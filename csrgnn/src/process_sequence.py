#!/usr/bin/env python
# coding: utf-8

from typing import List, Set
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
import random
import pickle
from rich import print
from sklearn.model_selection import KFold

from process_csv import analyze_trends, categorize_csv_features, categorize_layer3_features, categorize_layer4_features

tqdm.pandas()

def generate_sequence_pickle(observe_window: int = -1,
                             predict_window: List[int] = None,
                             remove_normal_nodes: bool = True,
                             add_trends: bool = False, 
                             add_layer3: bool = False,
                             add_layer4: bool = False,
                             negative_random_samples: str = None,
                             fold: int = 5, no_imputation: bool = False,
                             dataset_dir: Path = None, sched: int = 1):
    if predict_window is None:
        predict_window = [-1]
    elif not isinstance(predict_window, list):
        predict_window = [predict_window]
    assert isinstance(predict_window, list)
    if dataset_dir is None:
        dataset_dir = Path(__file__).resolve().parent.parent / 'datasets'
    if no_imputation:
        # 有缺失值的csv
        df_2012 = pd.read_csv(Path(__file__).resolve().parent.parent / 'datasets' / 'layers_2012_2019_preprocessed_noimputation.csv', index_col=0)
    else:
        df_2012 = pd.read_csv(Path(__file__).resolve().parent.parent / 'datasets' / 'layers_2012_2019_preprocessed.csv', index_col=0)

    categorize_csv_features(df_2012)
    if add_trends:
        analyze_trends(df_2012)
    if add_layer3:
        categorize_layer3_features(df_2012)
    if add_layer4:
        categorize_layer4_features(df_2012)

    sepsis_ids = df_2012[df_2012['infectionDay'].notna()]['id'].unique()
    print(f'{len(sepsis_ids)=}')
    unique_ids = df_2012['id'].unique()
    print(f'{len(unique_ids)=}')

    # train test split
    if fold == -1:
        all_patient_id = df_2012['id'].unique().tolist()
        random.seed(42)
        all_patient_id = sorted(all_patient_id)
        random.shuffle((all_patient_id))

        patient_id_train = all_patient_id[: int(len(all_patient_id) * 0.8)]
        patient_id_val = all_patient_id[int(len(all_patient_id) * 0.8) :]
        patient_id_val = set(patient_id_val)
    else:
        assert 1 <= fold <= 5
        unique_ids  = df_2012['id'].unique()
        unique_ids.sort()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        patient_id_5fold = []
        for train_index, test_index in kf.split(unique_ids):
            patient_id_5fold.append(unique_ids[test_index])
        # all_patient_id = set([d['patient_id'] for d in data])
        all_patient_id = set(unique_ids)
        patient_id_val = all_patient_id & set(patient_id_5fold[fold - 1])
        patient_id_train = all_patient_id - set(patient_id_5fold[fold - 1])

    print(f'{len(patient_id_train)=}')
    print(f'{len(patient_id_val)=}')


    data, all_node_names_in_sequences, max_concurrent_nodes_num = gen_sequences_from_df(df_2012, observe_window,
                                                                                        predict_window,
                                                                                        remove_normal_nodes,
                                                                                        add_trends, 
                                                                                        add_layer3, add_layer4,
                                                                                        negative_random_samples, 42,
                                                                                        patient_id_val)
    if sched >=2 :
        # sample more random negative windows
        data2, _all_node_names_in_sequences, _max_concurrent_nodes_num = gen_sequences_from_df(df_2012, observe_window,
                                                                                            predict_window,
                                                                                            remove_normal_nodes,
                                                                                            add_trends, 
                                                                                            add_layer3, add_layer4,
                                                                                            'nds',
                                                                                            random_state=9527,)
        data += data2
    if negative_random_samples == 'nds_head':
        data_head, _all_node_names_in_sequences, _max_concurrent_nodes_num = gen_sequences_from_df(df_2012, observe_window,
                                                                                            predict_window,
                                                                                            remove_normal_nodes,
                                                                                            add_trends, 
                                                                                            add_layer3, add_layer4,
                                                                                            'nds_head',
                                                                                            random_state=42,)
    print(f'{data[0]=}')

    # all_node_names = set(sum(sum([d['sequences'] for d in data], []), []))
    all_node_names = all_node_names_in_sequences
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
    


    data_train = [d for d in data if d['patient_id'] not in patient_id_val]
    if negative_random_samples != 'nds_head':
        data_val = [d for d in data if d['patient_id'] in patient_id_val]
    else:
        data_val = [d for d in data_head if d['patient_id'] in patient_id_val]

    (dataset_dir / 'raw').mkdir(parents=True, exist_ok=True)

    print('[b green]saving session sequences[/b green]')
    with open(dataset_dir / 'raw/sequences_dicts.pkl', 'wb') as fw:
        pickle.dump({'data_train': data_train,
                    'data_val': data_val,
                    'observe_window': observe_window,
                    'predict_window': predict_window}, fw)

    for sequences_dicts, save_fn in [(data_train, 'raw/train.txt'), (data_val, 'raw/test.txt')]:
        user_list, sequence_list, cue_l_list, y_l_list = stack_sequences(sequences_dicts, all_node_names_2_nid, observe_window, predict_window)
        if 'train' in save_fn:
            # up-sampling
            to_add = [(user, seq, cue_l, y_l) for user, seq, cue_l, y_l in zip(user_list, sequence_list, cue_l_list, y_l_list) if 1 in y_l]
            for user, seq, cue_l, y_l in to_add:
                user_list.extend([user] * 4)
                sequence_list.extend([seq] * 4)
                cue_l_list.extend([cue_l] * 4)
                y_l_list.extend([y_l] * 4)
            y_l_array = np.array(y_l_list)
            positive_seq_num = y_l_array[:, 0].sum()
            negative_seq_num = len(y_l_array) - positive_seq_num
        with open(dataset_dir / save_fn, 'wb') as fw:
            print(f'dump {save_fn} with {len(sequence_list)} sequences')
            print(f'#positive sequences={np.sum(y_l_list)}')
            pickle.dump((user_list, sequence_list, cue_l_list, y_l_list), fw)

    with open(dataset_dir / 'raw/node_count.txt', 'wb') as fw:
        pickle.dump({'node_count': len(all_node_names_2_nid),
                    'max_concurrent_nodes_num': max_concurrent_nodes_num,
                    'positive_seq_num': positive_seq_num,
                    'negative_seq_num': negative_seq_num}, fw)
    
    with open(dataset_dir / 'raw/all_node_names_2_nid.txt', 'wb') as fw:
        pickle.dump(all_node_names_2_nid, fw)


def stack_sequences(sequences_dicts, all_node_names_2_nid, observe_window: int, predict_window: List[int] = None):
    user_list, sequence_list, cue_l_list, y_l_list = [], [], [], []
    for d in tqdm(sequences_dicts):
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
    assert len(sequence_list) == len(sequences_dicts)
    return user_list, sequence_list, cue_l_list, y_l_list


def gen_sequences_from_df(df_2012: pd.DataFrame,
                          observe_window: int = -1,
                          predict_window: List[int] = None,
                          remove_normal_nodes: bool = True,
                          add_trends: bool = False, 
                          add_layer3: bool = False,
                          add_layer4: bool = False,
                          negative_random_samples: str =None,
                          random_state: int = 42,
                          patient_id_val: List[int] = None,):
    assert isinstance(predict_window, list)
    assert predict_window
    cat_features = ['hr_cat', 'sbp_cat', 'dbp_cat', 'map_cat', 'rr_cat', 'fio2_cat', 'temp_cat', 'bpGap_cat', 'bpHr_cat']
    if add_trends:
        cat_features += ['hr_trend_cat', 'sbp_trend_cat', 'dbp_trend_cat', 'map_trend_cat', 'rr_trend_cat', 'fio2_trend_cat', 'temp_trend_cat']
    if add_layer3:
        cat_features += ['bolus_cat', 'RBC_cat', 'surg_cat', ]
    if add_layer4:
        cat_features += ['bicarb_cat', 'StrongIon_cat', 'BunToCreatinine_cat', 'wbc_cat'] # + ['uop_trend_cat']
    print('Extracting sessions from csv')

    df_sepsis_patient_unique = df_2012[df_2012['infectionDay'].notna()][['id', 'infectionDay', 'infectionHour']].drop_duplicates()
    sepsis_happen_hours = (df_sepsis_patient_unique['infectionDay'] * 24 + df_sepsis_patient_unique['infectionHour']).values
    avg_sepsis_happen_hours = sepsis_happen_hours.mean()
    # average num of sequences for sepsis patients, including positive and negative sequences.
    avg_seq_num_of_sepsis_patients = avg_sepsis_happen_hours - 48 - observe_window

    # random select first N observation windows for non-sepsis patients
    # N is generated from norm distribution with same avg and std from sepsis happen time
    np.random.seed(random_state)
    # random_sample_nums = np.random.normal(avg_sepsis_happen_hours, sepsis_happen_hours.std(), len(df_2012['id'].unique()))
    random_sample_nums = np.random.uniform(size=len(df_2012['id'].unique()))
    patient_id_to_sample_nums = dict(zip(sorted(df_2012['id'].unique()), random_sample_nums))


    params = [(patient_id, _df, observe_window, predict_window, remove_normal_nodes, negative_random_samples,
               cat_features, patient_id_to_sample_nums[patient_id], patient_id in patient_id_val) for patient_id, _df in tqdm(df_2012.groupby('id'))]
    # results = [gen_sequences_from_one_patient(*param) for param in tqdm(params)]
    
    results = process_map(_gen_sequences_from_one_patient, params, max_workers=8, chunksize=8)
    data = []
    all_node_names_in_sequences: Set[str] = set()
    for max_concurrent_nodes_num, data_of_patient, _all_node_names_in_sequences in results:
        data.extend(data_of_patient)
        all_node_names_in_sequences.update(_all_node_names_in_sequences)
    return data, all_node_names_in_sequences, max_concurrent_nodes_num

def _gen_sequences_from_one_patient(p):
        return gen_sequences_from_one_patient(*p)

def gen_sequences_from_one_patient(patient_id, _df, observe_window, predict_window, remove_normal_nodes, negative_random_samples, cat_features, random_uniform_value, is_val):
    data_of_patient = []
    all_node_names_in_sequences: Set[str] = set()
    max_concurrent_nodes_num = 0
        # first_row = _df.iloc[0]
    sepsis_at_last = (_df.iloc[0]['Sepsis'] > 0)
    if observe_window == -1:
        seq = []
        sepsis_countdown_list = []
        for _, row in _df.iterrows():
            if row['sepsisOccurred'] == 1:
                break
            events: List[str] = []
            for feature_name in cat_features:
                feature_value = row[feature_name]
                assert isinstance(feature_value, str), f'Only support categorical features! ({feature_name}: {feature_value})'
                if remove_normal_nodes and 'normal' in feature_value:
                    continue
                events.append(f'{feature_name}:{feature_value}')
            if events:
                seq.append(events)
                all_node_names_in_sequences.update(events)
                max_concurrent_nodes_num = max(max_concurrent_nodes_num, len(events))
                sepsis_countdown_list.append(row['sepsisCountDown'])
        assert len(sepsis_countdown_list) == len(seq)
        observe_start_hour = _df.iloc[0]['day'] * 24 + _df.iloc[0]['hour']
        window = {
                'patient_id': patient_id,
                'sequences': seq,
                'sepsis_count_down': sepsis_countdown_list,  # sepsis counting down in hours, same length as seq
                'sepsis_at_last': sepsis_at_last,  # sepsis occurred at last
                'observe_start_hour': observe_start_hour,
            }
        data_of_patient.append(window)
    else:
        assert len(_df) > observe_window
        end_row_index = int(random_uniform_value * (len(_df) - observe_window) + observe_window)  # for negative sampling
        if patient_id==16245:
            print(f'{end_row_index=} ')
        if negative_random_samples == 'ous':
            # over-sampling positive and under-sampling negative
            np.random.seed(patient_id)
            NEG_SAMP_RATE = 0.2
            kept_negative_indices = np.random.choice(len(_df), int(len(_df) * NEG_SAMP_RATE), replace=False)
        for observe_start_row_index in range(len(_df)):
                # XXX 序列首尾处长度不如窗口大小时该怎么办。现在会有小于target长度的窗口
                # XXX 现在可能会有identical重复的序列（最严重是同一个人连续几个窗口都完全一样）。现在没管它，可能会增加负样本量
                # XXX 全部都正常的窗口是否可以删除？减少负样本量
            observe_start_hour = _df.iloc[observe_start_row_index]['day'] * 24 + _df.iloc[observe_start_row_index]['hour']
            if observe_start_hour + observe_window < 48:
                    # 收集48小时数据后再开始预测
                continue
            if observe_start_row_index + observe_window >= len(_df):
                    # 滑窗 右边缘 超过了这个患者的记录长度
                continue
            observe_window_df = _df.iloc[observe_start_row_index: observe_start_row_index + observe_window]  # sliding observation window
                # if patient_id==16245:
                #     print(f'1 {observe_start_hour=}, {observe_start_row_index=}, {observe_window_df["sepsisCountDown"].iloc[-1]}')
            if sepsis_at_last and observe_window_df['sepsisCountDown'].iloc[-1] <= 0:
                    # sepsis happened before last hour of observe window, which means observe window exceeded time of sepsis
                continue

            if (negative_random_samples == 'nds') and sepsis_at_last:
                # assert sepsis_countdown_list[-1] == observe_window_df['sepsisCountDown'].iloc[-1], f"{sepsis_countdown_list[-1]=}, {observe_window_df['sepsisCountDown'].iloc[-1]=}"
                # if sepsis_countdown_list[-1] > max(predict_window):
                if observe_window_df['sepsisCountDown'].iloc[-1] > max(predict_window):
                        # 有sepsis的患者只保留positive的训练数据
                    continue
                # if patient_id==16245:
                #     print(f'2 {observe_start_hour=}, {observe_start_row_index=}, ')
            if not sepsis_at_last and (negative_random_samples == 'nds'):
                    # only keep N=predict_window windows for non-sepsis patients
                    # if observe_start_hour + observe_window + max(predict_window) < random_uniform_value:
                    #     continue
                    # if observe_start_hour + observe_window > random_uniform_value:
                    #     continue
                if observe_start_row_index + observe_window + max(predict_window) < end_row_index:
                    continue
                if observe_start_row_index + observe_window > end_row_index:
                    continue
            elif not sepsis_at_last and (negative_random_samples == 'nds_head'):
                    # if observe_start_row_index + observe_window + max(predict_window) < end_row_index:
                    #     continue
                if observe_start_row_index + observe_window > end_row_index:
                    continue
            if negative_random_samples == 'ous' and (not is_val):
                # negative down-sampling
                if (not sepsis_at_last) or (sepsis_at_last and observe_window_df['sepsisCountDown'].iloc[-1] > max(predict_window)):
                    # observe_window的label是没有sepsis
                    if observe_start_row_index not in kept_negative_indices:
                        continue
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
                    if remove_normal_nodes and 'normal' in feature_value:
                        continue
                    events.append(f'{feature_name}:{feature_value}')
                if events:
                    seq.append(events)
                    all_node_names_in_sequences.update(events)
                    max_concurrent_nodes_num = max(max_concurrent_nodes_num, len(events))
                    sepsis_countdown_list.append(row['sepsisCountDown'])
            assert len(sepsis_countdown_list) == len(seq)
            if not seq:
                continue
                # if len(seq) != observe_window:
                #     # 在序列收尾，滑窗长度不足；或某时刻的值全部为空白
                #     # XXX 这里应该是可以分类讨论的。序列首尾、有缺失值应该都被允许
                #     continue
                # XXX 设计理念：没有ground truth，之后再生成
            
            window = {
                    'patient_id': patient_id,
                    'sequences': seq,
                    'sepsis_count_down': sepsis_countdown_list,  # sepsis counting down in hours, same length as seq
                    'sepsis_at_last': sepsis_at_last,  # sepsis occurred at last
                    'observe_start_hour': observe_start_hour,
                }
            data_of_patient.append(window)
        if sepsis_at_last and not data_of_patient:
                # 这个sepsis患者没有加数据进来
            if patient_id in {3502, 8175, 2993, 5566, 23372}:
                    # without imputation
                pass
            else:
                raise RuntimeError(f'这个sepsis患者没有加数据进来.{patient_id=}')
    return max_concurrent_nodes_num,data_of_patient, all_node_names_in_sequences
