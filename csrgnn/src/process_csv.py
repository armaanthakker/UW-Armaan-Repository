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


    def sepsis_occurred(row):
        if row['Sepsis'] == 0:
            return 0
        d, h = row['day'], row['hour']
        sd, sh = row['infectionDay'], row['infectionHour']
        # print(d, h, sd,sh)
        accu_hour = d*24 + h
        accu_sepsis_hour = sd*24 + sh
        return int(accu_hour >= accu_sepsis_hour)
    # 1 for sepsis happened before this hour, 0 otherwise
    df_2012['sepsisOccurred'] = df_2012.apply(sepsis_occurred, axis=1)


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
