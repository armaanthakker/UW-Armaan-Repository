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


def generate_sequence_pickle(observe_window: int = -1,
                             predict_window: List[int] = None,
                             remove_normal_noedes: bool =True):
    if predict_window is None:
        predict_window = [-1]
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

    print(f'{(all_node_names)=}')
    print(f'{len(all_node_names)=}')

    sepsis_seq = [d['sequences'] for d in data if d['sepsis']]
    no_sepsis_seq = [d['sequences'] for d in data if not d['sepsis']]


    # write data file
    all_patient_id = [d['id'] for d in data]
    random.seed(42)
    all_patient_id = sorted(all_patient_id)
    random.shuffle((all_patient_id))

    patient_id_train = all_patient_id[: int(len(all_patient_id) * 0.8)]
    patient_id_val = all_patient_id[int(len(all_patient_id) * 0.8) :]

    print(f'{len(patient_id_train)=}')
    print(f'{len(patient_id_val)=}')


    data_train = [d for d in data if d['id'] not in patient_id_val]
    data_val = [d for d in data if d['id'] in patient_id_val]


    all_node_names_2_nid = {n: idx+67 for idx, n in enumerate(all_node_names + ['sepsis', 'no_sepsis'])}
    print(f'{(all_node_names_2_nid)=}')


    print('[b green]saving session sequences[/b green]')
    for sequences_list, save_fn in [(data_train, 'raw/train.txt'), (data_val, 'raw/test.txt')]:
        user_list, sequence_list, cue_l_list, y_l_list = stack_sequences(sequences_list, all_node_names_2_nid, observe_window)
        with open(dataset_dir / save_fn, 'wb') as fw:
            pickle.dump((user_list, sequence_list, cue_l_list, y_l_list), fw)

    with open(dataset_dir / 'raw/node_count.txt', 'wb') as fw:
        pickle.dump(len(all_node_names_2_nid), fw)

def stack_sequences(sequences_list, all_node_names_2_nid, observe_window):
    user_list, sequence_list, cue_l_list, y_l_list = [], [], [], []
    for d in sequences_list:
        sequences = d['sequences']
        sequences_nid = [[all_node_names_2_nid[node] for node in events] for events in sequences]
        is_sepsis = d['sepsis']
        sepsis_count_down = d['sepsis_count_down']
        if observe_window == -1:
            if len(sequences) < 3:
                continue
            for end_idx in range(3, len(sequences) + 1):
                    # end_idx = 3, 4, ..., len(sequences)
                sub_seq = sequences_nid[: end_idx]
                user_list.append(str(d['id']))
                sequence_list.append(sub_seq)
                cue_l_list.append([all_node_names_2_nid['sepsis']])
                    # cue_l_list.append([all_node_names_2_nid['sepsis'], all_node_names_2_nid['no_sepsis']])
                y_l_list.append([int(is_sepsis)])
                    # y_l_list.append([int(is_sepsis), 1-int(is_sepsis)])
        else:
                #TODO use obs and pred window
            pass
    return user_list,sequence_list,cue_l_list,y_l_list


def gen_sequences_from_df(df_2012,
                          observe_window: int = -1,
                          predict_window: List[int] = None,
                          remove_normal_noedes: bool = True):
    cat_features = ['hr_cat', 'sbp_cat', 'map_cat', 'rr_cat', 'fio2_cat', 'temp_cat', 'bpGap_cat', 'bpHr_cat']
    data = []
    for id, _df in tqdm(df_2012.groupby('id')):
        latest_feats = {f: None for f in cat_features}
        # first_row = _df.iloc[0]
        sepsis_occur = not np.isnan(_df.iloc[0]['infectionDay'])
        seq = []
        sepsis_countdown_list = []
        for _, row in _df.iterrows():
            if row['sepsisOccured'] == 1:
                break
            events = []
            for feature_name in cat_features:
                feature_value = row[feature_name]
                if isinstance(feature_value, str):
                    # TODO: 允许自循环链接
                    if 'normal' not in feature_value and feature_value != latest_feats[feature_name]:
                        events.append(f'{feature_name}:{feature_value}')
                        latest_feats[feature_name] = feature_value
            if events:
                seq.append(events)
                sepsis_countdown_list.append(row['sepsisCountDown'])
        assert len(sepsis_countdown_list) == len(seq)
        data.append({
            'id': id,
            'sequences': seq,
            'sepsis_count_down': sepsis_countdown_list,  # sepsis counting down in hours, same length as seq
            'sepsis': sepsis_occur,  # sepsis occurred at last
        })
        
    return data
