# -*- coding: utf-8 -*-
"""
Descripttion: 
Author: SijinHuang
Date: 2021-12-21 06:56:45
LastEditors: SijinHuang
LastEditTime: 2022-03-19 11:06:34
"""
import copy
import os
import argparse
import logging
import time
import random
import pickle
import torch
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from rich import print as rprint
from process_sequence import generate_sequence_pickle
from dataset import MultiSessionsGraph
from torch_geometric.data import DataLoader
from model import GNNModel
from train import forward
from torch.utils.tensorboard import SummaryWriter

os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()
# Logger configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observe_window', type=int, default=12, help='observation window in hours')
    parser.add_argument('--predict_window', type=int, default=[6], nargs='+', help='prediction window in hours')
    parser.add_argument('--fold', type=int, default=1, help='5 fold index')
    parser.add_argument("--no_imputation", default=False, action='store_true', help='use raw data with missing values')
    parser.add_argument("--remove_normal_nodes", default=False, action='store_true', help='remove normal events in session graphs')
    parser.add_argument("--add_trends", default=False, action='store_true', help='add trends for vital signs in session graphs')
    parser.add_argument("--add_layer4", default=False, action='store_true', help='add layer 4 features')
    parser.add_argument("--nrs", default=False, action='store_true', help='negative_random_samples reduce num of session graphs for non-sepsis patients')
    parser.add_argument("--skip_preprocess", default=False, action='store_true', help='skip csv data preprocessing')
    parser.add_argument("--only_preprocess", default=False, action='store_true', help='only csv data preprocessing, skip training')


    parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
    parser.add_argument('--hidden_size', type=int, default=50, help='hidden state size')
    parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
    parser.add_argument('--lr_dc', type=float, default=0.5, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    parser.add_argument('--num_layers', type=int, default=1, help='the number convolution layers')
    parser.add_argument("--use_gat", default=False, action='store_true', help='use GAT layers')
    parser.add_argument("--use_san", default=False, action='store_true', help='use self attention layers')
    # parser.add_argument("--reprocess_csv", default=False, action='store_true', help='force reprocessing data')
    parser.add_argument("--ignore_yaml", default=False, action='store_true', help='ignore YAML configure file')
    opt = parser.parse_args()
    if not opt.ignore_yaml:
        with open('config.yml') as f:
            config_yml = yaml.safe_load(f)
        if config_yml:
            vars(opt).update(config_yml)
    dataset_desc = f'obs={opt.observe_window},pred={opt.predict_window},trend={opt.add_trends},l4={opt.add_layer4},'\
        f'negSamp={opt.nrs},impu={not opt.no_imputation},fold={opt.fold}'
    train_desc = f'layers={opt.num_layers},gat={opt.use_gat}'
    args_desc = f'{dataset_desc},{train_desc}'
    return opt, args_desc, dataset_desc

def main():
    args, args_desc, dataset_desc = parse_args()
    logging.warning(args)

    # RNN with CUDA has non-determinism issues
    # torch.manual_seed(0)
    # random.seed(0)
    # np.random.seed(0)

    dataset_dir_name = f'datasets_{dataset_desc}'
    print(f'{Path(__file__).resolve()=}')
    print(f'{Path(__file__).parent.parent.resolve()=}')
    dataset_dir = Path(__file__).resolve().parent.parent / dataset_dir_name
    # (dataset_dir / 'raw').mkdir(parents=True, exist_ok=True)
    # (dataset_dir / 'processed').mkdir(parents=True, exist_ok=True)
    if args.skip_preprocess:
        rprint(f'[bold red]Data preprocessing skipped. Make sure cached pickles match configs![/bold red]')
    else:
        generate_sequence_pickle(
            args.observe_window, args.predict_window,
            args.remove_normal_nodes,
            args.add_trends, args.add_layer4, args.nrs,
            args.fold, args.no_imputation,
            dataset_dir
        )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cur_dir = os.getcwd()
    train_dataset = MultiSessionsGraph(cur_dir + f'/../{dataset_dir_name}', phrase='train',
                                       skip_preprocess=args.skip_preprocess)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=8)
    test_dataset = MultiSessionsGraph(cur_dir + f'/../{dataset_dir_name}', phrase='test',
                                      skip_preprocess=args.skip_preprocess)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=8)
    print(f'{len(train_dataset)=}')
    print(f'{len(test_dataset)=}')
    if args.only_preprocess:
        rprint(f'[bold red]Finished data preprocessing. Exit...[/bold red]')
        exit()
    # raise RuntimeError()

    # log_dir = cur_dir + '/../log/' + str(args) + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # log_dir = log_dir.replace('Namespace', '').replace(' ', '')
    _log_dir_name = time.strftime("%m-%d %H:%M", time.localtime()) + '(' + args_desc + ')'
    _log_dir_name = _log_dir_name[:250]
    log_dir = cur_dir + '/../log/' + _log_dir_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.warning('logging to {}'.format(log_dir))
    writer = SummaryWriter(log_dir)
    # use markdown to save hyperparameters
    _rows = ['|hp|value|', '|-|-|'] + [f'|{k}|{v}|' for k, v in vars(args).items()]
    writer.add_text('params', '\n'.join(_rows))
    conf_path = cur_dir + '/../log_configs/' + _log_dir_name + '.yml'
    os.makedirs(os.path.dirname(conf_path), exist_ok=True)
    with open(conf_path, 'w') as fw:
        _args = copy.deepcopy(vars(args))
        _args['time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        yaml.safe_dump(_args, fw)

    node_count_dict = pickle.load(open(cur_dir + f'/../{dataset_dir_name}/raw/node_count.txt','rb'))
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
        writer.add_scalar('lr/lr', scheduler.get_last_lr()[0], epoch)
        with torch.no_grad():
            forward(model, test_loader, device, writer, epoch, train_flag=False, csv_metrics=csv_metrics)
        scheduler.step()

    model_path = log_dir.replace('log', 'model')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logging.warning('saving model to {}'.format(model_path))

    csv_path = cur_dir + '/../log_csv/' + _log_dir_name + '.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    _df = pd.DataFrame(csv_metrics)
    _df.to_csv(csv_path, index=False)
    logging.warning('saving csv log to {}'.format(csv_path))
    # writer.add_hparams({k: str(v) for k,v in vars(args).items()}, csv_metrics[-1])

    writer.close()

if __name__ == '__main__':
    main()
