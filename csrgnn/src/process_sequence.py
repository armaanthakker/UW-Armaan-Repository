#!/usr/bin/env python
# coding: utf-8

from typing import List
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import random
import pickle
from rich import print

from process_csv import categorize_csv_features

tqdm.pandas()

def generate_sequence_pickle(observe_window: int = -1,
                             predict_window: List[int] = None,
                             remove_normal_noedes: bool =True):
    if predict_window is None:
        predict_window = [-1]
    elif not isinstance(predict_window, list):
        predict_window = [predict_window]
    assert isinstance(predict_window, list)
    dataset_dir = (Path(__file__).parent / '../datasets').resolve()
    df_2012 = pd.read_csv(dataset_dir / 'layers_2012_2015_preprocessed.csv')
    df_2012.drop(columns='Unnamed: 0', inplace=True)

    categorize_csv_features(df_2012)

    sepsis_ids = df_2012[df_2012['infectionDay'].notna()]['id'].unique()
    print(f'{len(sepsis_ids)=}')
    unique_ids = df_2012['id'].unique()
    print(f'{len(unique_ids)=}')

    data = gen_sequences_from_df(df_2012, observe_window,
                                 predict_window,
                                 remove_normal_noedes)
    print(f'{data[0]=}')

    all_node_names = set(sum(sum([d['sequences'] for d in data], []), []))
    all_node_names = sorted(all_node_names)
    all_node_names += [f'sepsis_in_{predcit_hour}hours' for predcit_hour in predict_window if predcit_hour > 0]
    all_node_names += ['sepsis_at_last', 'no_sepsis_at_last']
    print(f'{(all_node_names)=}')
    print(f'{len(all_node_names)=}')
    all_node_names_2_nid = {n: idx+67 for idx, n in enumerate(all_node_names)}
    print(f'{(all_node_names_2_nid)=}')

    # sepsis_seq = [d['sequences'] for d in data if d['sepsis_at_last']]
    # no_sepsis_seq = [d['sequences'] for d in data if not d['sepsis_at_last']]

    # write data file
    all_patient_id = set([d['patient_id'] for d in data])
    random.seed(42)
    all_patient_id = sorted(all_patient_id)
    random.shuffle((all_patient_id))

    patient_id_train = all_patient_id[: int(len(all_patient_id) * 0.8)]
    patient_id_val = all_patient_id[int(len(all_patient_id) * 0.8) :]
    patient_id_val = set(patient_id_val)

    print(f'{len(patient_id_train)=}')
    print(f'{len(patient_id_val)=}')


    data_train = [d for d in data if d['patient_id'] not in patient_id_val]
    data_val = [d for d in data if d['patient_id'] in patient_id_val]



    print('[b green]saving session sequences[/b green]')
    for sequences_list, save_fn in [(data_train, 'raw/train.txt'), (data_val, 'raw/test.txt')]:
        user_list, sequence_list, cue_l_list, y_l_list = stack_sequences(sequences_list, all_node_names_2_nid, observe_window)
        with open(dataset_dir / save_fn, 'wb') as fw:
            pickle.dump((user_list, sequence_list, cue_l_list, y_l_list), fw)

    with open(dataset_dir / 'raw/node_count.txt', 'wb') as fw:
        pickle.dump(len(all_node_names_2_nid), fw)

def stack_sequences(sequences_list, all_node_names_2_nid, observe_window: int, predict_window: List[int] = None):
    user_list, sequence_list, cue_l_list, y_l_list = [], [], [], []
    for d in tqdm(sequences_list):
        sequences: List[List[str]] = d['sequences']
        sequences_nid = [[all_node_names_2_nid[node] for node in events] for events in sequences]  # 字符串转node index
        is_sepsis = d['sepsis_at_last']
        sepsis_count_down_list: List[int] = d['sepsis_count_down']
        if observe_window == -1:
            if len(sequences) < 3:
                continue
            for end_idx in range(3, len(sequences) + 1):
                    # end_idx = 3, 4, ..., len(sequences)
                sub_seq = sequences_nid[: end_idx]
                user_list.append(str(d['patient_id']))
                sequence_list.append(sub_seq)
                cue_l_list.append([all_node_names_2_nid['sepsis_at_last']])
                    # cue_l_list.append([all_node_names_2_nid['sepsis_at_last'], all_node_names_2_nid['no_sepsis_at_last']])
                y_l_list.append([int(is_sepsis)])
                    # y_l_list.append([int(is_sepsis), 1-int(is_sepsis)])
        else:
            user_list.append(str(d['patient_id']))
            sequence_list.append(sequences_nid)
            target_nodes = []
            target_nodes_label = []
            for predcit_hour in predict_window:
                if predcit_hour == -1:
                    target_nodes.append(all_node_names_2_nid['sepsis_at_last'])
                    target_nodes_label.append(int(is_sepsis))
                else:
                    target_nodes.append(all_node_names_2_nid[f'sepsis_in_{predcit_hour}hours'])
                    target_nodes_label.append(int(is_sepsis and (sepsis_count_down_list[-1] <= predcit_hour)))  # and sepsis_count_down_list[-1] != -1
            cue_l_list.append(target_nodes)
            y_l_list.append(target_nodes_label)
    return user_list, sequence_list, cue_l_list, y_l_list


def gen_sequences_from_df(df_2012: pd.DataFrame,
                          observe_window: int = -1,
                          predict_window: List[int] = None,
                          remove_normal_noedes: bool = True):
    # assert isinstance(predict_window, list)
    cat_features = ['hr_cat', 'sbp_cat', 'map_cat', 'rr_cat', 'fio2_cat', 'temp_cat', 'bpGap_cat', 'bpHr_cat']
    data = []
    for patient_id, _df in tqdm(df_2012.groupby('id')):
        # first_row = _df.iloc[0]
        sepsis_at_last = not np.isnan(_df.iloc[0]['infectionDay'])
        if observe_window == -1:
            seq = []
            sepsis_countdown_list = []
            for _, row in _df.iterrows():
                if row['sepsisOccurred'] == 1:
                    break
                events = []
                for feature_name in cat_features:
                    feature_value = row[feature_name]
                    assert isinstance(feature_value, str), f'Only support categorical features! ({feature_name}: {feature_value})'
                    if remove_normal_noedes and 'normal' in feature_value:
                        continue
                    events.append(f'{feature_name}:{feature_value}')
                if events:
                    seq.append(events)
                    sepsis_countdown_list.append(row['sepsisCountDown'])
            assert len(sepsis_countdown_list) == len(seq)
            data.append({
                'patient_id': patient_id,
                'sequences': seq,
                'sepsis_count_down': sepsis_countdown_list,  # sepsis counting down in hours, same length as seq
                'sepsis_at_last': sepsis_at_last,  # sepsis occurred at last
            })
        else:
            for observe_start_hour in range(len(_df)):
                # XXX 序列首尾处长度不如窗口大小时该怎么办。现在会有小于target长度的窗口
                # XXX 现在可能会有identical重复的序列（最严重是同一个人连续几个窗口都完全一样）。现在没管它，可能会增加负样本量
                # XXX 全部都正常的窗口是否可以删除？减少负样本量
                if observe_start_hour + observe_window < 48:
                    # 收集48小时数据后再开始预测
                    continue
                observe_window_df = _df.iloc[observe_start_hour: observe_start_hour + observe_window]  # sliding observation window
                seq = []
                sepsis_countdown_list = []
                for _, row in observe_window_df.iterrows():
                    if row['sepsisOccurred'] == 1:
                        # 只保留sepsis发生前的数据
                        break
                    events = []
                    for feature_name in cat_features:
                        feature_value = row[feature_name]
                        if pd.isna(feature_value):
                            # print(f'NaN value! ({feature_name}: {feature_value})')
                            continue
                        assert isinstance(feature_value, str), f'Only support categorical features! ({feature_name}: {feature_value})'
                        if remove_normal_noedes and 'normal' in feature_value:
                            continue
                        events.append(f'{feature_name}:{feature_value}')
                    if events:
                        seq.append(events)
                        sepsis_countdown_list.append(row['sepsisCountDown'])
                assert len(sepsis_countdown_list) == len(seq)
                if not seq:
                    continue
                # XXX 设计理念：没有ground truth，之后再生成
                data.append({
                    'patient_id': patient_id,
                    'sequences': seq,
                    'sepsis_count_down': sepsis_countdown_list,  # sepsis counting down in hours, same length as seq
                    'sepsis_at_last': sepsis_at_last,  # sepsis occurred at last
                })
            
    return data
