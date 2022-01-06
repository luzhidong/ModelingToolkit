# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:17:01 2018

@author: Zhidong
"""


import pandas as pd
#import numpy as np
import time
import multiprocessing

#import os
#import sys
from addr_pseg_cut4 import func_pseg_cut





def MultiProc_Main(in_df, cpu_cnt=2, chunk_size=10000):
    _time = time.time()
    print('-'*40)
    print('Multiple Processing Begin.')
    
    #####
    _max_cpu_cnt = multiprocessing.cpu_count()
    if cpu_cnt>_max_cpu_cnt:
        print('Error: "cpu_cnt" is set to {0}, but maximum is {1}'.format(cpu_cnt, _max_cpu_cnt))
        BaseException
    #####
    
    #####
    in_df = in_df.copy()
    in_df['_chunk_flag'] = [int(s0/chunk_size) for s0 in range(in_df.shape[0])]
    in_df_groupby = list(in_df.groupby('_chunk_flag'))
    #####
    
    print('passing time: {0}'.format(time.time()-_time))
    #####
    pool = multiprocessing.Pool(processes=cpu_cnt)
    MupliProc_Result = pool.map(MulitProc_Unit_Func, (s0[1] for s0 in in_df_groupby))
    pool.close()
    pool.join()
    
    out_df = pd.concat(MupliProc_Result)
    del out_df['_chunk_flag']
    #####
    
    print('Multiple Processing Finish.')
    print('passing time: {0}'.format(time.time()-_time))
    print('-'*40)
    
    return out_df



def MulitProc_Unit_Func(in_df):
    #in_df = in_df[in_df['address'].str.contains('广东')]
    #out_df = in_df
    print('!!!')
    #return out_df


df = df_address[:]


_time = time.time()
s = MultiProc_Main(df,cpu_cnt=4,chunk_size=100)
print(time.time()-_time)


_time = time.time()
s1 = df['address'].apply(func_pseg_cut)
print(time.time()-_time)
