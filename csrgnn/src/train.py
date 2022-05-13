# -*- coding: utf-8 -*-
"""
Descripttion: 
Author: SijinHuang
Date: 2022-02-01 01:29:23
LastEditors: SijinHuang
LastEditTime: 2022-05-13 02:24:20
"""
import torch
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix


def forward(model, loader, device, writer, epoch, optimizer=None, train_flag=True, csv_metrics=None):
    if train_flag:
        model.train()
        tag = 'train'
    else:
        model.eval()
        tag = 'test'

    # to collect per batch accuracy, precision, recall, f1score, auc
    # accuracy, precision, recall, f1score, auc = [], [], [], [], []
    total_loss = 0.0
    total_examples = 0.0
    updates_per_epoch = len(loader)

    y_test_all = []
    y_pred_prob_all = []

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
        # y_pred = torch.round(y_prob_tensor).detach().cpu().numpy()
        y_test = targets.detach().cpu().numpy()

        y_test_all.extend(y_test.tolist())  # OOM???
        y_pred_prob_all.extend(y_prob.tolist())

        total_loss += loss.item() * batch.num_graphs
        total_examples += batch.num_graphs
        # accuracy.append(accuracy_score(y_test, y_pred))
        # precision.append(precision_score(y_test, y_pred))
        # recall.append(recall_score(y_test, y_pred))
        # f1score.append(f1_score(y_test, y_pred))
        # try:
        #     auc.append(roc_auc_score(y_test, y_prob))
        # except ValueError:
        #     pass


    # mean_accuracy = np.mean(accuracy)
    # mean_precision = np.mean(precision)
    # mean_recall = np.mean(recall)
    # mean_f1score = np.mean(f1score)
    # mean_auc = np.mean(auc) 
    
    y_pred_prob_all = np.array(y_pred_prob_all)
    y_pred_all = (y_pred_prob_all > 0.5).astype(int)
    y_pred_all_20 = (y_pred_prob_all > 0.2).astype(int)

    mean_accuracy = accuracy_score(y_test_all, y_pred_all)
    mean_precision = precision_score(y_test_all, y_pred_all)
    mean_recall = recall_score(y_test_all, y_pred_all)
    mean_f1score = f1_score(y_test_all, y_pred_all)
    mean_auc = roc_auc_score(y_test_all, y_pred_prob_all)
    ap = average_precision_score(y_test_all, y_pred_prob_all)
    mean_sensitivity = mean_recall
    mean_specificity = recall_score(y_test_all, y_pred_all, pos_label=0)
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test_all, y_pred_all)

    mean_accuracy_20 = accuracy_score(y_test_all, y_pred_all_20)
    mean_precision_20 = precision_score(y_test_all, y_pred_all_20)
    mean_recall_20 = recall_score(y_test_all, y_pred_all_20)
    mean_f1score_20 = f1_score(y_test_all, y_pred_all_20)
    mean_sensitivity_20 = mean_recall_20
    mean_specificity_20 = recall_score(y_test_all, y_pred_all_20, pos_label=0)
    [[tn_20, fp_20], [fn_20, tp_20]] = confusion_matrix(y_test_all, y_pred_all_20)

    writer.add_scalar('loss/{}_loss'.format(tag), total_loss/total_examples, epoch)
    writer.add_scalar('{}/mean_accuracy'.format(tag), mean_accuracy, epoch)
    writer.add_scalar('{}/mean_precision'.format(tag), mean_precision, epoch)
    writer.add_scalar('{}/mean_recall'.format(tag), mean_recall, epoch)
    writer.add_scalar('{}/mean_f1score'.format(tag), mean_f1score, epoch)
    writer.add_scalar('{}/mean_auc'.format(tag), mean_auc, epoch)
    writer.add_scalar('{}/AP'.format(tag), ap, epoch)
    writer.add_scalar('{}/mean_sensitivity'.format(tag), mean_sensitivity, epoch)
    writer.add_scalar('{}/mean_specificity'.format(tag), mean_specificity, epoch)

    writer.add_scalar('{}/mean_accuracy_20'.format(tag), mean_accuracy_20, epoch)
    writer.add_scalar('{}/mean_precision_20'.format(tag), mean_precision_20, epoch)
    writer.add_scalar('{}/mean_recall_20'.format(tag), mean_recall_20, epoch)
    writer.add_scalar('{}/mean_f1score_20'.format(tag), mean_f1score_20, epoch)
    writer.add_scalar('{}/mean_sensitivity_20'.format(tag), mean_sensitivity_20, epoch)
    writer.add_scalar('{}/mean_specificity_20'.format(tag), mean_specificity_20, epoch)
    
    cm_dict = {
        f'{tag}/cm_tn': tn,
        f'{tag}/cm_fp': fp,
        f'{tag}/cm_fn': fn,
        f'{tag}/cm_tp': tp,
        f'{tag}/cm_tn_20': tn_20,
        f'{tag}/cm_fp_20': fp_20,
        f'{tag}/cm_fn_20': fn_20,
        f'{tag}/cm_tp_20': tp_20,
    }
    for k, v in cm_dict.items():
        writer.add_scalar(k, v, epoch)

    print(f"epoch={epoch}, mean_accuracy={mean_accuracy}, mean_precision={mean_precision}, \
            mean_recall={mean_recall}, mean_f1score={mean_f1score}, mean_auc={mean_auc}, \
            AP={ap}, \
            mean_sensitivity={mean_sensitivity}, mean_specificity={mean_specificity}")
    print(f"epoch={epoch}, mean_accuracy_20={mean_accuracy_20}, mean_precision_20={mean_precision_20}, \
            mean_recall_20={mean_recall_20}, mean_f1score_20={mean_f1score_20}, \
            mean_sensitivity_20={mean_sensitivity_20}, mean_specificity_20={mean_specificity_20}")
    print(cm_dict)
    if csv_metrics is not None:
        row = {
            'epoch': epoch,
            'loss/{}_loss'.format(tag): total_loss/total_examples,
            '{}/mean_accuracy'.format(tag): mean_accuracy,
            '{}/mean_precision'.format(tag): mean_precision,
            '{}/mean_recall'.format(tag): mean_recall,
            '{}/mean_f1score'.format(tag): mean_f1score, 
            '{}/mean_auc'.format(tag): mean_auc, 
            '{}/AP'.format(tag): ap, 
            '{}/mean_sensitivity'.format(tag): mean_sensitivity,
            '{}/mean_specificity'.format(tag): mean_specificity,

            '{}/mean_accuracy_20'.format(tag): mean_accuracy_20,
            '{}/mean_precision_20'.format(tag): mean_precision_20,
            '{}/mean_recall_20'.format(tag): mean_recall_20,
            '{}/mean_f1score_20'.format(tag): mean_f1score_20, 
            '{}/mean_sensitivity_20'.format(tag): mean_sensitivity_20,
            '{}/mean_specificity_20'.format(tag): mean_specificity_20,
        }
        row.update(cm_dict)
        if csv_metrics and csv_metrics[-1]['epoch'] == epoch:
            csv_metrics[-1].update(row)
        else:
            csv_metrics.append(row)

