import torch
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def forward(model, loader, device, writer, epoch, optimizer=None, train_flag=True):
    if train_flag:
        model.train()
        tag = 'train'
    else:
        model.eval()
        tag = 'test'

    # to collect per batch accuracy, precision, recall, f1score, auc
    accuracy, precision, recall, f1score, auc = [], [], [], [], []
    total_loss = 0.0
    total_examples = 0.0
    updates_per_epoch = len(loader)

    for i, batch in enumerate(tqdm(loader, desc=f'epoch{epoch}')):
        if train_flag:
            optimizer.zero_grad()
        scores, _ = model(batch.to(device), batch.cue.to(device), device)
        targets = batch.y
        # scores.shape = batch.y.shape = (batch_size,)
        # only one target node for each graph/sequence.
        loss = model.loss_function(scores, targets.float())

        if train_flag:
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/train_batch_loss', loss.item(), epoch * updates_per_epoch + i)
        
        y_prob_tensor = torch.sigmoid(scores)
        y_prob = y_prob_tensor.detach().cpu().numpy()
        y_pred = torch.round(y_prob_tensor).detach().cpu().numpy()
        y_test = targets.detach().cpu().numpy()

        total_loss += loss.item() * batch.num_graphs
        total_examples += batch.num_graphs
        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        f1score.append(f1_score(y_test, y_pred))
        try:
            auc.append(roc_auc_score(y_test, y_prob))
        except ValueError:
            pass

    mean_accuracy = np.mean(accuracy)
    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    mean_f1score = np.mean(f1score)
    mean_auc = np.mean(auc) 
     
    writer.add_scalar('loss/{}_loss'.format(tag), total_loss/total_examples, epoch)
    writer.add_scalar('{}/mean_accuracy'.format(tag), mean_accuracy, epoch)
    writer.add_scalar('{}/mean_precision'.format(tag), mean_precision, epoch)
    writer.add_scalar('{}/mean_recall'.format(tag), mean_recall, epoch)
    writer.add_scalar('{}/mean_f1score'.format(tag), mean_f1score, epoch)
    writer.add_scalar('{}/mean_auc'.format(tag), mean_auc, epoch)
    print(f"epoch={epoch}, mean_accuracy={mean_accuracy}, mean_precision={mean_precision}, \
            mean_recall={mean_recall}, mean_f1score={mean_f1score}, mean_auc={mean_auc}")
