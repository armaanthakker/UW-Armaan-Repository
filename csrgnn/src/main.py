# -*- coding: utf-8 -*-
"""
Descripttion: 
Author: SijinHuang
Date: 2021-12-21 06:56:45
LastEditors: SijinHuang
LastEditTime: 2022-03-08 02:26:31
"""
import os
import argparse
import logging
import time
import pickle
import torch
import yaml
import pandas as pd
from tqdm import tqdm
from process_sequence import generate_sequence_pickle
from dataset import MultiSessionsGraph
from torch_geometric.data import DataLoader
from model import GNNModel
from train import forward
from torch.utils.tensorboard import SummaryWriter

# Logger configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observe_window', type=int, default=12, help='observation window in hours')
    parser.add_argument('--predict_window', type=int, default=[6], nargs='+', help='prediction window in hours')
    parser.add_argument("--remove_normal_nodes", default=False, action='store_true', help='remove normal events in session graphs')
    parser.add_argument("--add_trends", default=False, action='store_true', help='add trends for vital signs in session graphs')
    parser.add_argument("--nrs", default=False, action='store_true', help='negative_random_samples reduce num of session graphs for non-sepsis patients')

    parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
    parser.add_argument('--hidden_size', type=int, default=50, help='hidden state size')
    parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
    parser.add_argument('--lr_dc', type=float, default=0.5, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    parser.add_argument('--num_layers', type=int, default=1, help='the number convolution layers')
    parser.add_argument("--use_san", default=False, action='store_true', help='use self attention layers')
    parser.add_argument("--use_gat", default=False, action='store_true', help='use GAT layers')
    # parser.add_argument("--reprocess_csv", default=False, action='store_true', help='force reprocessing data')
    opt = parser.parse_args()
    with open('config.yml') as f:
        config_yml = yaml.safe_load(f)
    if config_yml:
        vars(opt).update(config_yml)
    return opt

def main():
    args = parse_args()
    logging.warning(args)
    generate_sequence_pickle(args.observe_window,
                             args.predict_window,
                             args.remove_normal_nodes,
                             args.add_trends,
                             args.nrs,)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cur_dir = os.getcwd()
    train_dataset = MultiSessionsGraph(cur_dir + '/../datasets', phrase='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = MultiSessionsGraph(cur_dir + '/../datasets', phrase='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f'{len(train_dataset)=}')
    print(f'{len(test_dataset)=}')
    # raise RuntimeError()

    log_dir = cur_dir + '/../log/' + str(args) + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_dir = log_dir.replace('Namespace', '').replace(' ', '')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.warning('logging to {}'.format(log_dir))
    writer = SummaryWriter(log_dir)

    node_count_dict = pickle.load(open(cur_dir + '/../datasets/raw/node_count.txt','rb'))
    n_node = node_count_dict['node_count']
    max_concurrent_nodes_num = node_count_dict['max_concurrent_nodes_num']

    model = GNNModel(hidden_size=args.hidden_size, n_node=n_node, num_layers=args.num_layers, use_san=args.use_san, use_gat=args.use_gat).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    logging.warning(model)
    # csv writer
    csv_metrics = []
    for epoch in tqdm(range(args.epoch)):
        print(f'Epoch: {epoch} LR: {scheduler.get_last_lr()[0]:E} {scheduler.get_last_lr()}') 
        forward(model, train_loader, device, writer, epoch, optimizer=optimizer, train_flag=True, csv_metrics=csv_metrics)
        scheduler.step()
        with torch.no_grad():
            forward(model, test_loader, device, writer, epoch, train_flag=False, csv_metrics=csv_metrics)

    model_path = log_dir.replace('log', 'model')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logging.warning('saving model to {}'.format(model_path))

    csv_path = log_dir.replace('log', 'log_csv') + '.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    _df = pd.DataFrame(csv_metrics)
    _df.to_csv(csv_path, index=False)
    logging.warning('saving csv log to {}'.format(csv_path))

    writer.close()

if __name__ == '__main__':
    main()
