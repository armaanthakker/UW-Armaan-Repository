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



def categorize_csv_features(df_2012: pd.DataFrame) -> pd.DataFrame:
    """categorize numeric features and adding category columns

    Args:
        df_2012 (pd.DataFrame): Origin numeric CSV

    Returns:
        pd.DataFrame: new DataFrame with categoy features
    """

    def cat_map_hr(row):
        hr = row['hr']
        if pd.isna(hr):
            return np.nan
        if hr > 120:
            return '>120'
        elif hr > 110:
            return '111-120'
        elif hr >= 100:
            return '100-110'
        else:
            return '<100(normal)'

    df_2012['hr_cat'] = df_2012.apply(cat_map_hr, axis=1)

    def cat_map_sbp(row):
        sbp = row['sbp']
        if pd.isna(sbp):
            return np.nan
        if sbp < 90:
            return '<90(very low)'
        elif sbp <= 100:
            return '90-100(low)'
        else:
            return '>100(normal)'

    df_2012['sbp_cat'] = df_2012.apply(cat_map_sbp, axis=1)


    def cat_map_map(row):
        val = row['map']
        if pd.isna(val):
            return np.nan
        if val < 65:
            return '<65'

        elif val <= 70:
            return '65-70'

        else:
            return '>70(normal)'

    df_2012['map_cat'] = df_2012.apply(cat_map_map, axis=1)


    def cat_map_rr(row):
        val = row['rr']
        if pd.isna(val):
            return np.nan
        if val > 24:
            return '>24'
        elif val >= 22:
            return '22-24'
        elif val >= 12:
            return '12-21(normal)'
        else:
            return '<12'
        
    df_2012['rr_cat'] = df_2012.apply(cat_map_rr, axis=1)


    # def cat_map_spo2(row):
    #     val = row['spo2']
    #     if pd.isna(val):
    #         return np.nan
    #     if val < 90:
    #         return '<90'
    #     elif val <= 92:
    #         return '91-92'
    #     else:
    #         return '>92(normal)'
        
    # df_2012['spo2_cat'] = df_2012.apply(cat_map_spo2, axis=1)


    def cat_map_fio2(row):
        val = row['fio2']
        if pd.isna(val):
            return np.nan
        if val > 80:
            return '>80'
        elif val >= 60:
            return '60-80'
        elif val >= 30:
            return '30-60'
        else:
            return '<30(normal)'
        
    df_2012['fio2_cat'] = df_2012.apply(cat_map_fio2, axis=1)


    def cat_map_temp(row):
        val = row['temp']
        if pd.isna(val):
            return np.nan
        if val > 38:
            return '>38(high)'
        elif val >= 36:
            return '36-38(normal)'
        else:
            return '<36(low)'
        
    df_2012['temp_cat'] = df_2012.apply(cat_map_temp, axis=1)


    def cat_map_bpGap(row):
        val = row['sbp'] - row['dbp']
        if pd.isna(val):
            return np.nan
        if val < 30:
            return '<30'
        else:
            return '>=30(normal)'
        
    df_2012['bpGap_cat'] = df_2012.apply(cat_map_bpGap, axis=1)


    def cat_map_bpHr(row):
        val = row['sbp'] / row['hr']
        if pd.isna(val):
            return np.nan
        if val < 1.1:
            return '<1.1'
        else:
            return '>=1.1(normal)'
        
    df_2012['bpHr_cat'] = df_2012.apply(cat_map_bpHr, axis=1)


    def sepsis_occured(row):
        if row['Sepsis'] == 0:
            return 0
        d, h = row['day'], row['hour']
        sd, sh = row['infectionDay'], row['infectionHour']
        # print(d, h, sd,sh)
        accu_hour = d*24 + h
        accu_sepsis_hour = sd*24 + sh
        return int(accu_hour >= accu_sepsis_hour)
    # 1 for sepsis happened before this hour, 0 otherwise
    df_2012['sepsisOccured'] = df_2012.apply(sepsis_occured, axis=1)


    def sepsis_count_down(row):
        if row['Sepsis'] == 0:
            return -1
        d, h = row['day'], row['hour']
        sd, sh = row['infectionDay'], row['infectionHour']
        # print(d, h, sd,sh)
        accu_hour = d*24 + h
        accu_sepsis_hour = sd*24 + sh
        return (accu_sepsis_hour - accu_hour)
    # count down hours to sepsis
    df_2012['sepsisCountDown'] = df_2012.apply(sepsis_count_down, axis=1)


    return df_2012


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

    data = split_by_patient_id(df_2012)        
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
        
        with open(dataset_dir / save_fn, 'wb') as fw:
            pickle.dump((user_list, sequence_list, cue_l_list, y_l_list), fw)

    with open(dataset_dir / 'raw/node_count.txt', 'wb') as fw:
        pickle.dump(len(all_node_names_2_nid), fw)


def split_by_patient_id(df_2012):
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
