# -*- coding: utf-8 -*-
# 
#----------------------------------------------------------
# script name: 
#----------------------------------------------------------
# creator: zhidong.lu
# create date: 2021-12-30
# update date: 2022-01-02
# version: 0.2
#----------------------------------------------------------
# 
# 
#----------------------------------------------------------


import os
import re
import sys
import csv
import json
import time
import pytz
import datetime
from collections import OrderedDict
from itertools import product
import pickle

import gc
import multiprocessing

import pandas as pd
import numpy as np
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

import random


_cwd = os.getcwd()
try:
    os.chdir(os.path.dirname(__file__))
except:
    pass
from FeatureEngineering import func_woe_report_v1
os.chdir(_cwd)


pd.pandas.set_option("display.max_columns", None)
pd.pandas.set_option("display.max_rows", 300)

plt.style.use({"figure.figsize": [s0*3 for s0 in (4, 2)]})
sns.set(style="whitegrid", rc={"figure.figsize": [s0*3 for s0 in (4, 2)]})



#################################################################
class BoostrapSamplingEstimator_statistics(object):
    """
    data                 > 
    epochs               >                Default: 1000
    batch_pct            > (0, 1]         Default: 0.5
    batch_size           >                Default: None
    sample_replace       > True/False     Default: False
    method               > mean/std/var   Default: mean
    reduce_udf           >                Default: None
    alpha                > (0, 1)         Default: 0.05
    lower_q              > (0, 1)         Default: None
    upper_q              > (0, 1)         Default: None
    random_seed          >                Default: None
    result_lst_cnt       >                Default: 1000
    """
    
    ############################################################
    def __init__(
        self,
        data,
        epochs=1000,
        batch_pct=0.5,
        batch_size=None,
        sample_replace=False,
        method="mean",
        reduce_udf=None,
        alpha=0.05,
        lower_q=None,
        upper_q=None,
        random_seed=None,
        result_lst_cnt=1000,
        ):
        
        #####################
        # initial
        #####################
        self.data = pd.Series(data).dropna().values
        self.data_size = self.data.shape[0]
        
        self.epochs = epochs
        
        self.batch_pct = batch_pct
        self.batch_size = batch_size
        if self.batch_size is None and (self.batch_pct>0 and self.batch_pct<=1):
            self.batch_size = int(self.data_size*self.batch_pct)
        
        self.sample_replace = sample_replace
        if self.sample_replace==False:
            self.batch_size = min(self.batch_size, self.data_size)
        
        self.method = method
        self.reduce_udf = reduce_udf
        
        self.reduce_func = None
        if self.reduce_udf is None:
            if self.method=="mean":
                self.reduce_func = np.mean
            elif self.method=="std":
                self.reduce_func = np.std
            elif self.method=="vat":
                self.reduce_func = np.var
            else:
                pass
        else:
            self.reduce_func = self.reduce_udf
        
        self.alpha = alpha
        self.lower_q = lower_q
        self.upper_q = upper_q
        if (self.alpha>0 and self.alpha<1) and (self.lower_q is None or self.upper_q is None):
            self.lower_q = self.alpha/2
            self.upper_q = 1-self.alpha/2
        elif (self.lower_q>0 and self.lower_q<1) and (self.upper_q>0 and self.upper_q<1):
            pass
        else:
            self.lower_q = 0.025
            self.upper_q = 0.975
        
        self.random_seed = random_seed
        self._random_seed = None
        
        self.result_lst_cnt = (0 if result_lst_cnt is None else result_lst_cnt)
        
        #####################
        self.fit_count = 0
        self.lower_ci = -np.inf
        self.upper_ci = np.inf
        self.ci = [self.lower_ci, self.upper_ci]
        self.ci_cnt = 0
        
        self.lower_ci_lst = -np.inf
        self.upper_ci_lst = np.inf
        self.ci_lst = [self.lower_ci_lst, self.upper_ci_lst]
        
        #####################
        self._sample = None
        self._result = None
        
        self.result_list_lst = list()
        self.result_list = list()
        self.result = None
    
    ############################################################
    def compute(
        self,
        with_return=False,
        ):
        
        #####################
        # compute
        #####################
        
        # sampling
        self.sampling()
        
        # 计算当前抽样下 result
        self._result = self.reduce_func(self._sample)
        self.result_list.append(self._result)
        self.result_list_lst = self.result_list_lst[-(self.result_lst_cnt-1):]+[self._result]
        
        # 更新 self.fit_count
        self.fit_count = self.fit_count+1
        
        # 更新 self.result
        self.update_result(_result=self._result)
        
        if with_return:
            return self.result
    
    ############################################################
    def compute_epochs(
        self,
        epochs=None,
        print_log_epochs=None,
        ):
        
        #####################
        # compute epochs
        #####################
        for _idx in range((self.epochs if epochs is None else epochs)):
            self.compute(with_return=False)
            
            if (print_log_epochs is not None) and (np.mod(self.fit_count, print_log_epochs)==0):
                print("fit_count: {}, result: {}".format(self.fit_count, self.result))
    
    ############################################################
    def sampling(
        self,
        ):
        
        #####################
        # sampling
        #####################
        if self.random_seed is not None:
            try:
                np.random.seed(seed=self.random_seed+self.fit_count)
                self._random_seed = self.random_seed+self.fit_count
            except ValueError:
                self.random_seed = np.mod(self.random_seed, 2**32-1)
        else:
            self.random_seed = np.random.randint(1000000)
            np.random.seed(seed=self.random_seed+self.fit_count)
            self._random_seed = self.random_seed+self.fit_count
        
        self._sample = np.random.choice(
            a=self.data,
            size=self.batch_size,
            replace=self.sample_replace,
        )
    
    ############################################################
    def update_result(
        self,
        _result,
        ):
        
        #####################
        # update result
        #####################
        # 更新 self.result
        if self.result is None:
            self.result = _result
        else:
            self.result = self.result+(_result-self.result)/self.fit_count
    
    ############################################################
    def calc_confidence_interval(
        self,
        result_lst_cnt=None,
        ):
        
        if result_lst_cnt is None:
            result_lst_cnt = self.result_lst_cnt
        
        #####################
        # calculate confidence interval
        #####################
        if len(self.result_list)>0:
            self.lower_ci, self.upper_ci = np.quantile(
                a=self.result_list,
                q=[self.lower_q, self.upper_q],
                interpolation="linear",
            ).tolist()
            self.ci = [self.lower_ci, self.upper_ci]
            
            self.ci_cnt = len(self.result_list)
        
        if len(self.result_list_lst)>0 and result_lst_cnt>0:
            if self.fit_count%result_lst_cnt==0:
                self.lower_ci_lst, self.upper_ci_lst = np.quantile(
                    a=self.result_list_lst[-result_lst_cnt:],
                    q=[self.lower_q, self.upper_q],
                    interpolation="linear",
                ).tolist()
                self.ci_lst = [self.lower_ci_lst, self.upper_ci_lst]
        elif len(self.result_list_lst)>0 and result_lst_cnt==-1:
            self.lower_ci_lst, self.upper_ci_lst = np.quantile(
                a=self.result_list_lst[:],
                q=[self.lower_q, self.upper_q],
                interpolation="linear",
            ).tolist()
            self.ci_lst = [self.lower_ci_lst, self.upper_ci_lst]
    
    ############################################################
    def clear_result_list(
        self,
        ):
        
        self.result_list = list()
        _ = gc.collect()
    
    ############################################################
    def main(
        self,
        calc_confidence_interval=True,
        clear_result_list=True,
        epochs=None,
        print_log_epochs=None,
        ):
        
        self.compute_epochs(epochs=epochs, print_log_epochs=print_log_epochs)
        
        if calc_confidence_interval==True:
            self.calc_confidence_interval(result_lst_cnt=self.result_lst_cnt)
        
        if clear_result_list==True:
            self.clear_result_list()
    
    
#################################################################
class BoostrapSamplingEstimator_percentile(BoostrapSamplingEstimator_statistics):
    """
    data                 > 
    q                    > [0, 100]
    epochs               >                Default: 1000
    batch_pct            > (0, 1]         Default: 0.5
    batch_size           >                Default: None
    sample_replace       > True/False     Default: False
    alpha                > (0, 1)         Default: 0.05
    lower_q              > (0, 1)         Default: None
    upper_q              > (0, 1)         Default: None
    random_seed          >                Default: None
    result_lst_cnt       >                Default: 1000
    """
    
    ############################################################
    def __init__(
        self,
        data,
        q,
        epochs=1000,
        batch_pct=0.5,
        batch_size=None,
        sample_replace=False,
        alpha=0.05,
        lower_q=None,
        upper_q=None,
        random_seed=None,
        result_lst_cnt=1000,
        ):
        
        #####################
        # initial
        #####################
        self.q = q
        
        #####################
        # super initial
        #####################
        super(BoostrapSamplingEstimator_percentile, self).__init__(
            data=data,
            epochs=epochs,
            batch_pct=batch_pct,
            batch_size=batch_size,
            sample_replace=sample_replace,
            method=None,
            reduce_udf=lambda s0: np.percentile(s0, q=q, interpolation="linear"),
            alpha=alpha,
            lower_q=lower_q,
            upper_q=upper_q,
            random_seed=random_seed,
            result_lst_cnt=result_lst_cnt,
        )
    
    
#################################################################
class BoostrapSamplingEstimator_woe(BoostrapSamplingEstimator_statistics):
    """
    in_var               > 
    in_target            > 
    epochs               >                Default: 1000
    batch_pct            > (0, 1]         Default: 0.5
    batch_size           >                Default: None
    sample_replace       > True/False     Default: False
    alpha                > (0, 1)         Default: 0.05
    lower_q              > (0, 1)         Default: None
    upper_q              > (0, 1)         Default: None
    random_seed          >                Default: None
    result_lst_cnt       >                Default: 100
    
    ----------------------------------------------------------
    like `func_woe_report_v1`
    ----------------------------------------------------------
    """
    
    ############################################################
    def __init__(
        self,
        in_var,
        in_target,
        epochs=100,
        batch_pct=0.5,
        batch_size=None,
        sample_replace=False,
        alpha=0.05,
        lower_q=None,
        upper_q=None,
        random_seed=None,
        result_lst_cnt=100,
        ):
        
        #####################
        # initial
        #####################
        self.in_var = pd.Series(in_var)
        self.in_target = pd.Series(in_target)
        self._var_label = self.in_var.sort_values().unique().tolist()
        self._target_label = self.in_target.sort_values().unique().tolist()
        
        #####################
        # super initial
        #####################
        super(BoostrapSamplingEstimator_woe, self).__init__(
            data=zip(in_var, in_target),
            epochs=epochs,
            batch_pct=batch_pct,
            batch_size=batch_size,
            sample_replace=sample_replace,
            method=None,
            reduce_udf=lambda s0: func_woe_report_v1(
                                        in_var=pd.Series((t[0] for t in s0)),
                                        in_target=pd.Series((t[1] for t in s0)),
                                        with_total=False, good_label_val="0_good", bad_label_val="1_bad",
                                        floating_point=1e-4,
                                        with_lift_ks=False, lift_calc_ascending=True,
                                        with_wls_adj_woe=False,
                                    )["WOE"].to_dict(),
            alpha=alpha,
            lower_q=lower_q,
            upper_q=upper_q,
            random_seed=random_seed,
            result_lst_cnt=result_lst_cnt,
        )
    
        #####################
        self.lower_ci = None
        self.upper_ci = None
        self.ci = None
        self.lower_ci_lst = None
        self.upper_ci_lst = None
        self.ci_lst = None
        
    ############################################################
    def compute_epochs(
        self,
        epochs=None,
        print_log_epochs=None,
        ):
        
        #####################
        # compute epochs
        #####################
        for _idx in range((self.epochs if epochs is None else epochs)):
            self.compute(with_return=False)
            
            if (print_log_epochs is not None) and (np.mod(self.fit_count, print_log_epochs)==0):
                print("fit_count: {}, result: {}".format(self.fit_count, list(self.result.values())))
    
    ############################################################
    def sampling(
        self,
        ):
        
        #####################
        # sampling
        #####################
        if self.random_seed is not None:
            try:
                np.random.seed(seed=self.random_seed+self.fit_count)
                self._random_seed = self.random_seed+self.fit_count
            except ValueError:
                self.random_seed = np.mod(self.random_seed, 2**32-1)
        else:
            self.random_seed = np.random.randint(1000000)
            np.random.seed(seed=self.random_seed+self.fit_count)
            self._random_seed = self.random_seed+self.fit_count
        
        self._sample = np.concatenate([
                np.random.choice(
                    a=pd.Series((s0 for s0 in self.data if s0[1]==self._target_label[0])),
                    size=1,
                ),
                np.random.choice(
                    a=pd.Series((s0 for s0 in self.data if s0[1]==self._target_label[1])),
                    size=1,
                ),
                np.random.choice(
                    a=self.data,
                    size=self.batch_size-2,
                    replace=self.sample_replace,
                ),
            ],
            axis=0,
        )
    
    ############################################################
    def update_result(
        self,
        _result,
        ):
        
        #####################
        # update result
        #####################
        # 更新 self.result
        if self.result is None:
            self.result = _result
        else:
            self.result = \
                (
                    pd.Series(self.result)+
                    (pd.Series(_result)-pd.Series(self.result))/self.fit_count
                ).to_dict()
    
    ############################################################
    def calc_confidence_interval(
        self,
        result_lst_cnt=None,
        ):
        
        if result_lst_cnt is None:
            result_lst_cnt = self.result_lst_cnt
        
        #####################
        # calculate confidence interval
        #####################
        if len(self.result_list)>0:
            self.ci = pd.DataFrame(self.result_list).T.apply(
                lambda s0: np.quantile(
                                a=s0,
                                q=[self.lower_q, self.upper_q],
                                interpolation="linear",
                            ),
                axis=1,
            ).to_dict()
            self.lower_ci = pd.Series(self.ci).apply(lambda s0: s0[0]).to_dict()
            self.upper_ci = pd.Series(self.ci).apply(lambda s0: s0[1]).to_dict()
            
            self.ci_cnt = len(self.result_list)
        
        if len(self.result_list_lst)>0 and result_lst_cnt>0:
            if self.fit_count%result_lst_cnt==0:
                self.ci_lst = pd.DataFrame(self.result_list_lst[-result_lst_cnt:]).T.apply(
                    lambda s0: np.quantile(
                                    a=s0,
                                    q=[self.lower_q, self.upper_q],
                                    interpolation="linear",
                                ),
                    axis=1,
                ).to_dict()
                self.lower_ci_lst = pd.Series(self.ci_lst).apply(lambda s0: s0[0]).to_dict()
                self.upper_ci_lst = pd.Series(self.ci_lst).apply(lambda s0: s0[1]).to_dict()
        elif len(self.result_list_lst)>0 and result_lst_cnt==-1:
            self.ci_lst = pd.DataFrame(self.result_list_lst[:]).T.apply(
                lambda s0: np.quantile(
                                a=s0,
                                q=[self.lower_q, self.upper_q],
                                interpolation="linear",
                            ),
                axis=1,
            ).to_dict()
            self.lower_ci_lst = pd.Series(self.ci_lst).apply(lambda s0: s0[0]).to_dict()
            self.upper_ci_lst = pd.Series(self.ci_lst).apply(lambda s0: s0[1]).to_dict()
    
    
#################################################################




if __name__=="__main__":
    pass

