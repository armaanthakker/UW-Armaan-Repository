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

# tqdm.pandas()

from pandarallel import pandarallel

pandarallel.initialize(nb_workers=8, progress_bar=False)


def categorize_csv_features(df_2012: pd.DataFrame) -> pd.DataFrame:
    """categorize numeric features and adding category columns

    Args:
        df_2012 (pd.DataFrame): Origin numeric CSV

    Returns:
        pd.DataFrame: new DataFrame with categoy features
    """
    print('[b yellow]Start adding category columns[/b yellow]')

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

    # df_2012['hr_cat'] = df_2012.progress_apply(cat_map_hr, axis=1)
    df_2012['hr_cat'] = df_2012.parallel_apply(cat_map_hr, axis=1)

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

    # df_2012['sbp_cat'] = df_2012.progress_apply(cat_map_sbp, axis=1)
    df_2012['sbp_cat'] = df_2012.parallel_apply(cat_map_sbp, axis=1)


    def cat_map_dbp(row):
        val = row['dbp']
        if pd.isna(val):
            return np.nan
        if val < 60:
            return '<60'
        else:
            return '>=60(normal)'

    # df_2012['dbp_cat'] = df_2012.progress_apply(cat_map_dbp, axis=1)
    df_2012['dbp_cat'] = df_2012.parallel_apply(cat_map_dbp, axis=1)


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

    # df_2012['map_cat'] = df_2012.progress_apply(cat_map_map, axis=1)
    df_2012['map_cat'] = df_2012.parallel_apply(cat_map_map, axis=1)


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
        
    # df_2012['rr_cat'] = df_2012.progress_apply(cat_map_rr, axis=1)
    df_2012['rr_cat'] = df_2012.parallel_apply(cat_map_rr, axis=1)


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
        
    # # df_2012['spo2_cat'] = df_2012.apply(cat_map_spo2, axis=1)
    # df_2012['spo2_cat'] = df_2012.parallel_apply(cat_map_spo2, axis=1)


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
        
    # df_2012['fio2_cat'] = df_2012.progress_apply(cat_map_fio2, axis=1)
    df_2012['fio2_cat'] = df_2012.parallel_apply(cat_map_fio2, axis=1)


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
        
    # df_2012['temp_cat'] = df_2012.progress_apply(cat_map_temp, axis=1)
    df_2012['temp_cat'] = df_2012.parallel_apply(cat_map_temp, axis=1)


    df_2012['bpGap'] = df_2012['sbp'] - df_2012['dbp']

    def cat_map_bpGap(row):
        val = row['bpGap']
        if pd.isna(val):
            return np.nan
        if val < 30:
            return '<30'
        else:
            return '>=30(normal)'
        
    # df_2012['bpGap_cat'] = df_2012.progress_apply(cat_map_bpGap, axis=1)
    df_2012['bpGap_cat'] = df_2012.parallel_apply(cat_map_bpGap, axis=1)


    df_2012['bpHr'] = df_2012['sbp'] / df_2012['hr']

    def cat_map_bpHr(row):
        val = row['bpHr']
        if pd.isna(val):
            return np.nan
        if val < 1.1:
            return '<1.1'
        else:
            return '>=1.1(normal)'
        
    # df_2012['bpHr_cat'] = df_2012.progress_apply(cat_map_bpHr, axis=1)
    df_2012['bpHr_cat'] = df_2012.parallel_apply(cat_map_bpHr, axis=1)

    df_2012['accu_hour'] = df_2012['day'] * 24 + df_2012['hour']

    def sepsis_occurred(row):
        if row['Sepsis'] == 0:
            return 0
        sd, sh = row['infectionDay'], row['infectionHour']
        accu_hour = row['accu_hour']
        accu_sepsis_hour = sd*24 + sh
        return int(accu_hour >= accu_sepsis_hour)
    # 1 for sepsis happened before this hour, 0 otherwise
    # df_2012['sepsisOccurred'] = df_2012.progress_apply(sepsis_occurred, axis=1)
    df_2012['sepsisOccurred'] = df_2012.parallel_apply(sepsis_occurred, axis=1)


    def sepsis_count_down(row):
        if row['Sepsis'] == 0:
            return -1
        sd, sh = row['infectionDay'], row['infectionHour']
        accu_hour = row['accu_hour']
        accu_sepsis_hour = sd*24 + sh
        return (accu_sepsis_hour - accu_hour)
    # count down hours to sepsis
    # df_2012['sepsisCountDown'] = df_2012.progress_apply(sepsis_count_down, axis=1)
    df_2012['sepsisCountDown'] = df_2012.parallel_apply(sepsis_count_down, axis=1)


    return df_2012


def analyze_trends(df_2012: pd.DataFrame) -> pd.DataFrame:
    """adding columns for trends of each vital signs

    Args:
        df_2012 (pd.DataFrame): Origin numeric CSV

    Returns:
        pd.DataFrame: new DataFrame with trends values
    """
    print('[b yellow]Start adding trends columns[/b yellow]')

    window_length = 6
    def rolling_mean_previous(df, column, previous_window = 6):
        # rolling average of previous 6 hours
        # https://stackoverflow.com/questions/48967165/using-shift-and-rolling-in-pandas-with-groupby
        return df[column].groupby(df['id']).shift(1).rolling(previous_window).mean()
    
    df_2012['hr_rolling_avg'] = rolling_mean_previous(df_2012, 'hr', window_length)
    df_2012['sbp_rolling_avg'] = rolling_mean_previous(df_2012, 'sbp', window_length)
    df_2012['dbp_rolling_avg'] = rolling_mean_previous(df_2012, 'dbp', window_length)
    df_2012['map_rolling_avg'] = rolling_mean_previous(df_2012, 'map', window_length)
    df_2012['rr_rolling_avg'] = rolling_mean_previous(df_2012, 'rr', window_length)
    df_2012['fio2_rolling_avg'] = rolling_mean_previous(df_2012, 'fio2', window_length)
    df_2012['temp_rolling_avg'] = rolling_mean_previous(df_2012, 'temp', window_length)

    def cat_map_hr_trend(row):
        val_prev = row['hr_rolling_avg']
        val = row['hr']
        if pd.isna(val) or pd.isna(val_prev):
            return np.nan
        if val - val_prev > 5:
            return "Increase"
        else:
            return "normal"
    # df_2012['hr_trend_cat'] = df_2012.progress_apply(cat_map_hr_trend, axis=1)
    df_2012['hr_trend_cat'] = df_2012.parallel_apply(cat_map_hr_trend, axis=1)


    def cat_map_sbp_trend(row):
        val_prev = row['sbp_rolling_avg']
        val = row['sbp']
        if pd.isna(val) or pd.isna(val_prev):
            return np.nan
        if val - val_prev > 5:
            return "Increase"
        else:
            return "normal"
    # df_2012['sbp_trend_cat'] = df_2012.progress_apply(cat_map_sbp_trend, axis=1)
    df_2012['sbp_trend_cat'] = df_2012.parallel_apply(cat_map_sbp_trend, axis=1)


    def cat_map_dbp_trend(row):
        val_prev = row['dbp_rolling_avg']
        val = row['dbp']
        if pd.isna(val) or pd.isna(val_prev):
            return np.nan
        if val - val_prev > 5:
            return "Increase"
        else:
            return "normal"
    # df_2012['dbp_trend_cat'] = df_2012.progress_apply(cat_map_dbp_trend, axis=1)
    df_2012['dbp_trend_cat'] = df_2012.parallel_apply(cat_map_dbp_trend, axis=1)


    def cat_map_map_trend(row):
        val_prev = row['map_rolling_avg']
        val = row['map']
        if pd.isna(val) or pd.isna(val_prev):
            return np.nan
        if val - val_prev > 5:
            return "Increase"
        else:
            return "normal"
    # df_2012['map_trend_cat'] = df_2012.progress_apply(cat_map_map_trend, axis=1)
    df_2012['map_trend_cat'] = df_2012.parallel_apply(cat_map_map_trend, axis=1)

    def cat_map_rr_trend(row):
        val_prev = row['rr_rolling_avg']
        val = row['rr']
        if pd.isna(val) or pd.isna(val_prev):
            return np.nan
        if val - val_prev > 2.5:
            return "Increase"
        else:
            return "normal"
    # df_2012['rr_trend_cat'] = df_2012.progress_apply(cat_map_rr_trend, axis=1)
    df_2012['rr_trend_cat'] = df_2012.parallel_apply(cat_map_rr_trend, axis=1)

    def cat_map_fio2_trend(row):
        val_prev = row['fio2_rolling_avg']
        val = row['fio2']
        if pd.isna(val) or pd.isna(val_prev):
            return np.nan
        if val - val_prev > 5:
            return "Increase"
        else:
            return "normal"
    # df_2012['fio2_trend_cat'] = df_2012.progress_apply(cat_map_fio2_trend, axis=1)
    df_2012['fio2_trend_cat'] = df_2012.parallel_apply(cat_map_fio2_trend, axis=1)

    def cat_map_temp_trend(row):
        val_prev = row['temp_rolling_avg']
        val = row['temp']
        if pd.isna(val) or pd.isna(val_prev):
            return np.nan
        def _get_temp_ordinal(val):
            if val > 38:
                # return '>38(high)'
                return 3
            elif val >= 36:
                # return '36-38(normal)'
                return 2
            else:
                # return '<36(low)'
                return 1
        temp_ordinal = _get_temp_ordinal(val)
        temp_prev_ordinal = _get_temp_ordinal(val_prev)
        if temp_ordinal > temp_prev_ordinal:
            return "Increase"
        elif temp_ordinal < temp_prev_ordinal:
            return "Decrease"
        else:
            return "Sustain(normal)"
    # df_2012['temp_trend_cat'] = df_2012.progress_apply(cat_map_temp_trend, axis=1)
    df_2012['temp_trend_cat'] = df_2012.parallel_apply(cat_map_temp_trend, axis=1)

    return df_2012


def categorize_layer4_features(df_2012: pd.DataFrame) -> pd.DataFrame:
    """categorize numeric layer 4 features and adding category columns

    Args:
        df_2012 (pd.DataFrame): Origin numeric CSV

    Returns:
        pd.DataFrame: new DataFrame with categoy features
    """
    print('[b yellow]Start adding category Layer 4 columns[/b yellow]')

    def cat_map_bicarb(row):
        # could directly use `acidosisCat` column.
        val = row['bicarb']
        if pd.isna(val):
            return np.nan
        if val > 22:
            return '>22(1)'
        elif val >= 20:
            return '20-22(2)'
        elif val >= 17:
            return '17-19(bad)'
        else:
            return '<17(very bad)'
    # df_2012['bicarb_cat'] = df_2012.progress_apply(cat_map_bicarb, axis=1)
    df_2012['bicarb_cat'] = df_2012.parallel_apply(cat_map_bicarb, axis=1)


    def cat_map_StrongIon(row):
        # could directly use `acidosisCat` column.
        val = row['StrongIon']
        if pd.isna(val):
            return np.nan
        if val > 22:
            return '>22(1)'
        elif val >= 20:
            return '20-22(2)'
        elif val >= 17:
            return '17-19(bad)'
        else:
            return '<17(very bad)'
    # df_2012['StrongIon_cat'] = df_2012.progress_apply(cat_map_StrongIon, axis=1)
    df_2012['StrongIon_cat'] = df_2012.parallel_apply(cat_map_StrongIon, axis=1)


    def cat_map_BunToCreatinine(row):
        val = row['bun'] / row['creatinine']
        if pd.isna(val):
            return np.nan
        if val < 5:
            return '<5(low)'
        elif val <= 20:
            return '5-20 (normal)'
        elif val <= 40:
            return '20-40 (elevated)'
        else:
            return '>40 (high)'
    # df_2012['BunToCreatinine_cat'] = df_2012.progress_apply(cat_map_BunToCreatinine, axis=1)
    df_2012['BunToCreatinine_cat'] = df_2012.parallel_apply(cat_map_BunToCreatinine, axis=1)


    def cat_map_wbc(row):
        val = row['wbc']
        if pd.isna(val):
            return np.nan
        if val > 14:
            return '>14(high)'
        elif val >= 12:
            return '12-14(elevated)'
        elif val >= 4:
            return '4-11(normal)'
        else:
            return '<4 (low - also not good)'
    # df_2012['wbc_cat'] = df_2012.progress_apply(cat_map_wbc, axis=1)
    df_2012['wbc_cat'] = df_2012.parallel_apply(cat_map_wbc, axis=1)

    # window_length = 6
    # def rolling_mean_previous(df, column, previous_window = 6):
    #     # rolling average of previous 6 hours
    #     # https://stackoverflow.com/questions/48967165/using-shift-and-rolling-in-pandas-with-groupby
    #     return df[column].groupby(df['id']).shift(1).rolling(previous_window).mean()
    
    # df_2012['uop_rolling_avg'] = rolling_mean_previous(df_2012, 'uop', window_length)
    # def cat_map_uop_trend(row):
    #     val_prev = row['uop_rolling_avg']
    #     val = row['uop']
    #     if pd.isna(val) or pd.isna(val_prev):
    #         return np.nan
    #     if val - val_prev < 40:
    #         return "Decrease"
    #     else:
    #         return "normal"
    # # df_2012['uop_trend_cat'] = df_2012.progress_apply(cat_map_uop_trend, axis=1)
    # df_2012['uop_trend_cat'] = df_2012.parallel_apply(cat_map_uop_trend, axis=1)

    return df_2012


def categorize_layer3_features(df_2012: pd.DataFrame) -> pd.DataFrame:
    """categorize numeric layer 3 features and adding category columns

    Args:
        df_2012 (pd.DataFrame): Origin numeric CSV

    Returns:
        pd.DataFrame: new DataFrame with categoy features
    """
    print('[b yellow]Start adding category Layer 3 columns[/b yellow]')

    def hourly_delta(df, column):
        # delta value of column from previous row
        return df[column] - df[column].groupby(df['id']).shift(1, fill_value=0)  # output origin value at first hour

    df_2012['bolus_delta'] = hourly_delta(df_2012, 'bolusSum')
    df_2012.loc[df_2012['bolus_delta'] > 0, 'bolus_cat'] = 'bolus'
    df_2012['RBC_delta'] = hourly_delta(df_2012, 'RBCsum')
    df_2012.loc[df_2012['RBC_delta'] > 0, 'RBC_cat'] = 'RBC'
    df_2012['surg_delta'] = hourly_delta(df_2012, 'surgSum')
    df_2012.loc[df_2012['surg_delta'] > 0, 'surg_cat'] = 'surg'

    return df_2012
