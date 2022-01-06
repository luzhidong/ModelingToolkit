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

import tensorflow as tf
import tensorflow.keras as keras
# import tensorflow.experimental.numpy as tnp
import tensorflow_probability as tfp


_cwd = os.getcwd()
try:
    os.chdir(os.path.dirname(__file__))
except:
    pass
from FeatureEngineering import func_binning_continuous_v1
os.chdir(_cwd)


pd.pandas.set_option("display.max_columns", None)
pd.pandas.set_option("display.max_rows", 300)

plt.style.use({"figure.figsize": [s0*3 for s0 in (4, 2)]})
sns.set(style="whitegrid", rc={"figure.figsize": [s0*3 for s0 in (4, 2)]})



#################################################################
# layer
# boostrap 计算分位数
class TFLayer_boostrap_calc_percentile(tf.keras.layers.Layer):
    
    """
    
    ----------------------------------------------------------
    like `BoostrapSamplingEstimator_percentile`
    ----------------------------------------------------------
    """
    
    ############################################################
    def __init__(
        self,
        q,
        trainable=False,
        output_with_onehot_trans=False,
        right_border=True,
        include_lowest=False,
        out_type="03_onehot_vector",
        alpha=0.05,
        lower_q=None,
        upper_q=None,
        result_lst_cnt=None,
        with_update_result=True,
        with_calc_confidence_interval=True,
        dtype=tf.float64,
        ):
        
        #####################
        # initial
        #####################
        self.q = q
        self.trainable = trainable
        self.output_with_onehot_trans = output_with_onehot_trans
        self.right_border = right_border
        self.include_lowest = include_lowest
        self.out_type = out_type
        
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
        
        self.result_lst_cnt = (0 if result_lst_cnt is None else result_lst_cnt)
        
        self.with_update_result = with_update_result
        self.with_calc_confidence_interval = with_calc_confidence_interval
        
        #####################
        self.fit_count = 0
        self.fit_samp_count_avg = None
        
        self.lower_ci_lst = None
        self.upper_ci_lst = None
        self.ci_lst = None
        
        #####################
        self._result = None
        
        self.result_list_lst = list()
        self.result = None
        self.cut_point = None
        
        self._layer_info_data = None
        
        #####################
        # super initial
        #####################
        super(TFLayer_boostrap_calc_percentile, self).__init__(
            dtype=dtype,
        )
    
    # ############################################################
    # def build(
    #     self,
    #     input_shape,
    #     ):
    #     pass
    
    def _func_tfp_percentile(self, x, q):
        # 针对 x 全为 nan 时的处理
        rt = tf.cond(
            pred=(x.shape[0]!=0),
            true_fn=lambda: tfp.stats.percentile(
                x=x, q=q,
                axis=0,
                interpolation="linear",
            ),
            false_fn=lambda: tf.constant([np.NaN for s0 in q], tf.float64),
        )
        return rt
    
    ############################################################
    def call(
        self,
        inputs,
        ):
        
        if self.with_update_result==True:
            #####################
            # self._result = tf.map_fn(
            #     fn=lambda s0: tfp.stats.percentile(
            #         x=tf.gather(
            #             params=s0,
            #             indices=tf.where(tf.math.logical_not(tf.math.is_nan(s0)))[:, 0],
            #         ),
            #         q=self.q,
            #         axis=0,
            #         interpolation="linear",
            #     ),
            #     elems=tf.transpose(inputs, perm=[1, 0]),
            # )
            self._result = tf.map_fn(
                fn=lambda s0: self._func_tfp_percentile(
                    x=tf.gather(
                        params=s0,
                        indices=tf.where(tf.math.logical_not(tf.math.is_nan(s0)))[:, 0],
                    ),
                    q=self.q,
                ),
                elems=tf.transpose(inputs, perm=[1, 0]),
            )
            self.result_list_lst = self.result_list_lst[-(self.result_lst_cnt-1):]+[self._result]

            # 更新 self.fit_count
            self.fit_count = self.fit_count+1

            # 更新 self.result
            self.update_result(_result=self._result)
            self.cut_point = self.result
            
            # 更新 self.fit_samp_count_avg
            if self.fit_samp_count_avg is None:
                self.fit_samp_count_avg = inputs.shape[0]
            else:
                self.fit_samp_count_avg = self.fit_samp_count_avg + \
                    (inputs.shape[0]-self.fit_samp_count_avg)/self.fit_count
            
        else:
            pass
        
        if self.with_update_result==True and self.with_calc_confidence_interval==True:
            # 计算 self.ci_lst
            self.calc_confidence_interval(result_lst_cnt=self.result_lst_cnt)
        else:
            pass
        
        if self.output_with_onehot_trans==True and self.result is not None:
            rt = func_tf_binning_continous(
                tensor_inputs=inputs,
                tensor_cut_point=self.cut_point,
                out_type=self.out_type,
                right_border=self.right_border, include_lowest=self.include_lowest,
            )
            return rt
        else:
            return self.result
    
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
            # 处理 _result 和 self.result 中的 nan
            _result = tf.where(condition=tf.math.is_nan(_result), x=self.result, y=_result)
            self.result = tf.where(condition=tf.math.is_nan(self.result), x=_result, y=self.result)
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
        if len(self.result_list_lst)>0 and result_lst_cnt>0:
            if self.fit_count%result_lst_cnt==0:
                self.ci_lst = self.ci_lst = tf.transpose(
                    a=np.quantile(
                        a=tf.concat(
                            values=[
                                tf.expand_dims(s0, axis=0)
                                for s0 in self.result_list_lst[-result_lst_cnt:]
                            ],
                            axis=0,
                        ).numpy(),
                        q=[self.lower_q, self.upper_q],
                        axis=0,
                        interpolation="linear",
                    ),
                    perm=[1, 2, 0],
                )
                self.lower_ci_lst = self.ci_lst[:, :, 0]
                self.upper_ci_lst = self.ci_lst[:, :, 1]
        elif len(self.result_list_lst)>0 and result_lst_cnt==-1:
            self.ci_lst = self.ci_lst = tf.transpose(
                a=np.quantile(
                    a=tf.concat(
                        values=[
                            tf.expand_dims(s0, axis=0)
                            for s0 in self.result_list_lst[:]
                        ],
                        axis=0,
                    ).numpy(),
                    q=[self.lower_q, self.upper_q],
                    axis=0,
                    interpolation="linear",
                ),
                perm=[1, 2, 0],
            )
            self.lower_ci_lst = self.ci_lst[:, :, 0]
            self.upper_ci_lst = self.ci_lst[:, :, 1]
    
    ############################################################
    def clear_result_list_lst(
        self,
        ):
        
        self.result_list_lst = list()
        # _ = gc.collect()
    
    ############################################################
    def write_layer_info_data(
        self,
        file=None,
        ):
        
        self._layer_info_data = OrderedDict({
            "q": list(self.q),
            "trainable": self.trainable,
            "output_with_onehot_trans": self.output_with_onehot_trans,
            "right_border": self.right_border,
            "include_lowest": self.include_lowest,
            "out_type": self.out_type,
            "alpha": self.alpha,
            "lower_q": self.lower_q,
            "upper_q": self.upper_q,
            "result_lst_cnt": self.result_lst_cnt,
            "with_update_result": self.with_update_result,
            "with_calc_confidence_interval": self.with_calc_confidence_interval,
            "fit_count": self.fit_count,
            "fit_samp_count_avg": self.fit_samp_count_avg,
            "lower_ci_lst": (self.lower_ci_lst.numpy() if self.lower_ci_lst is not None else None),
            "upper_ci_lst": (self.upper_ci_lst.numpy() if self.upper_ci_lst is not None else None),
            "ci_lst": (self.ci_lst.numpy() if self.ci_lst is not None else None),
            "_result": self._result.numpy(),
            "result_list_lst": [s0.numpy() for s0 in self.result_list_lst],
            "result": self.result.numpy(),
            "cut_point": self.cut_point.numpy(),
        })
        
        if file is not None:
            with open(file=file, mode="wb") as fw:
                pickle.dump(self._layer_info_data, file=fw)
        
        self._layer_info_data = None
    
    ############################################################
    def read_layer_info_data(
        self,
        file,
        ):
        
        with open(file=file, mode="rb") as fr:
            self._layer_info_data = pickle.load(file=fr)

        self.trainable = self._layer_info_data["trainable"]
        self.output_with_onehot_trans = self._layer_info_data["output_with_onehot_trans"]
        self.right_border = self._layer_info_data["right_border"]
        self.include_lowest = self._layer_info_data["include_lowest"]
        self.out_type = self._layer_info_data["out_type"]
        self.alpha = self._layer_info_data["alpha"]
        self.lower_q = self._layer_info_data["lower_q"]
        self.upper_q = self._layer_info_data["upper_q"]
        self.result_lst_cnt = self._layer_info_data["result_lst_cnt"]
        self.with_update_result = self._layer_info_data["with_update_result"]
        self.with_calc_confidence_interval = self._layer_info_data["with_calc_confidence_interval"]
        self.fit_count = self._layer_info_data["fit_count"]
        self.fit_samp_count_avg = self._layer_info_data["fit_samp_count_avg"]
        self.lower_ci_lst = (tf.constant(self._layer_info_data["lower_ci_lst"]) if self._layer_info_data["lower_ci_lst"] is not None else None)
        self.upper_ci_lst = (tf.constant(self._layer_info_data["upper_ci_lst"]) if self._layer_info_data["upper_ci_lst"] is not None else None)
        self.ci_lst = (tf.constant(self._layer_info_data["ci_lst"]) if self._layer_info_data["ci_lst"] is not None else None)
        self._result = tf.constant(self._layer_info_data["_result"])
        self.result_list_lst = [tf.constant(s0) for s0 in self._layer_info_data["result_list_lst"]]
        self.result = tf.constant(self._layer_info_data["result"])
        self.cut_point = tf.constant(self._layer_info_data["cut_point"])
        
        self._layer_info_data = None
    
    
#################################################################
# layer
# boostrap 计算woe
class TFLayer_boostrap_calc_woe(tf.keras.layers.Layer):
    
    """
    
    ----------------------------------------------------------
    like `BoostrapSamplingEstimator_woe`
    ----------------------------------------------------------
    """
    
    ############################################################
    def __init__(
        self,
        in_type="03_onehot_vector",
        output_with_woe_trans=True,
        with_wls_adj=False,
        output_with_wls_adj_details=False,
        trainable=False,
        alpha=0.05,
        lower_q=None,
        upper_q=None,
        result_lst_cnt=None,
        with_update_result=True,
        with_calc_confidence_interval=True,
        dtype=tf.float64,
        ):
        
        #####################
        # initial
        #####################
        self.in_type = in_type
        self.output_with_woe_trans = output_with_woe_trans
        self.with_wls_adj = with_wls_adj
        self.output_with_wls_adj_details = output_with_wls_adj_details
        
        self.trainable = trainable
        
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
        
        self.result_lst_cnt = (0 if result_lst_cnt is None else result_lst_cnt)
        
        self.with_update_result = with_update_result
        self.with_calc_confidence_interval = with_calc_confidence_interval
        
        #####################
        self.fit_count = 0
        self.fit_samp_count_avg = None
        
        self.lower_ci_lst = None
        self.upper_ci_lst = None
        self.ci_lst = None
        
        #####################
        self._tensor_woe = None
        self._tensor_wls_adj_woe = None
        self._dict_wls_adj_woe_details = None
        self.tensor_woe_list_lst = list()
        self.tensor_wls_adj_woe_list_lst = list()
        self.dict_wls_adj_woe_details_list_lst = list()
        self.tensor_woe = None
        self.tensor_wls_adj_woe = None
        
        self._layer_info_data = None
        
        #####################
        # super initial
        #####################
        super(TFLayer_boostrap_calc_woe, self).__init__(
            dtype=dtype,
        )
    
    # ############################################################
    # def build(
    #     self,
    #     input_shape,
    #     ):
    #     pass
    
    ############################################################
    def call(
        self,
        tensor_X_onehot_vector,
        tensor_y,
        ):
        
        if self.with_update_result==True:
            #####################
            if self.in_type=="02_onehot_class":
                tensor_X_onehot_vector = tf.transpose(
                    a=tf.cast(
                        x=tf.map_fn(
                            fn=lambda s0: tf.one_hot(
                                indices=s0,
                                depth=tf.reduce_max(tensor_X_onehot_vector, axis=(0, 1)).numpy()+1,
                                dtype=tf.int64,
                            ),
                            elems=tf.transpose(tf.cast(tensor_X_onehot_vector, dtype=tf.int64)),
                        ),
                        dtype=tf.int8,
                    ),
                    perm=[1, 0, 2],
                )

            #####################
            self._tensor_woe, self._tensor_wls_adj_woe, self._dict_wls_adj_woe_details = \
                func_tf_calc_woe(
                    tensor_X_onehot_vector=tensor_X_onehot_vector,
                    tensor_y=tensor_y,
                    with_wls_adj=self.with_wls_adj,
                    output_with_wls_adj_details=self.output_with_wls_adj_details,
                    floating_point=1e-4,
                )
            self.tensor_woe_list_lst = \
                self.tensor_woe_list_lst[-(self.result_lst_cnt-1):] + \
                [self._tensor_woe]
            self.tensor_wls_adj_woe_list_lst = \
                self.tensor_wls_adj_woe_list_lst[-(self.result_lst_cnt-1):] + \
                ([] if self._tensor_wls_adj_woe is None else [self._tensor_wls_adj_woe])
            self.dict_wls_adj_woe_details_list_lst = \
                self.dict_wls_adj_woe_details_list_lst[-(self.result_lst_cnt-1):] + \
                ([] if self._dict_wls_adj_woe_details is None else [self._dict_wls_adj_woe_details])

            # 更新 self.fit_count
            self.fit_count = self.fit_count+1

            # 更新 self.tensor_woe  self.tensor_wls_adj_woe
            self.update_result(_woe=[self._tensor_woe, self._tensor_wls_adj_woe])
            
            # 更新 self.fit_samp_count_avg
            if self.fit_samp_count_avg is None:
                self.fit_samp_count_avg = tensor_X_onehot_vector.shape[0]
            else:
                self.fit_samp_count_avg = self.fit_samp_count_avg + \
                    (tensor_X_onehot_vector.shape[0]-self.fit_samp_count_avg)/self.fit_count
            
        else:
            pass
        
        if self.with_update_result==True and self.with_calc_confidence_interval==True:
            # 计算 self.ci_lst
            self.calc_confidence_interval(result_lst_cnt=self.result_lst_cnt)
        else:
            pass
        
        if self.output_with_woe_trans==True:
            _tensor_woe = (
                self.tensor_wls_adj_woe
                if self.with_wls_adj==True else
                self.tensor_woe
            )
            
            if _tensor_woe is not None:
                rt = tf.reduce_sum(
                    input_tensor=tf.where(
                        condition=tensor_X_onehot_vector==1,
                        x=tf.where(
                            tf.math.is_nan(_tensor_woe),
                            x=0,
                            y=_tensor_woe,
                        ),
                        y=tf.zeros_like(_tensor_woe),
                    ),
                    axis=2,
                )
                return rt
            else:
                return tensor_X_onehot_vector
        else:
            return tensor_X_onehot_vector
        
    ############################################################
    def update_result(
        self,
        _woe,
        ):
        
        _tensor_woe, _tensor_wls_adj_woe = _woe
        
        #####################
        # update result
        #####################
        
        # 更新 self.tensor_woe
        if self.tensor_woe is None:
            self.tensor_woe = _tensor_woe
        else:
            # 处理 _tensor_woe 和 self.tensor_woe 中的 nan
            _tensor_woe = tf.where(condition=tf.math.is_nan(_tensor_woe), x=self.tensor_woe, y=_tensor_woe)
            self.tensor_woe = tf.where(condition=tf.math.is_nan(self.tensor_woe), x=_tensor_woe, y=self.tensor_woe)
            self.tensor_woe = \
                self.tensor_woe + \
                (_tensor_woe-self.tensor_woe)/self.fit_count
        
        # 更新 self.tensor_wls_adj_woe
        if self.tensor_wls_adj_woe is None:
            self.tensor_wls_adj_woe = _tensor_wls_adj_woe
        else:
            # 处理 _tensor_wls_adj_woe 和 self.tensor_wls_adj_woe 中的 nan
            _tensor_wls_adj_woe = tf.where(
                condition=tf.math.is_nan(_tensor_wls_adj_woe),
                x=self.tensor_wls_adj_woe, y=_tensor_wls_adj_woe,
            )
            self.tensor_wls_adj_woe = tf.where(
                condition=tf.math.is_nan(self.tensor_wls_adj_woe),
                x=_tensor_wls_adj_woe, y=self.tensor_wls_adj_woe,
            )
            self.tensor_wls_adj_woe = \
                self.tensor_wls_adj_woe + \
                (_tensor_wls_adj_woe-self.tensor_wls_adj_woe)/self.fit_count
    
    ############################################################
    def calc_confidence_interval(
        self,
        result_lst_cnt=None,
        ):
        
        if result_lst_cnt is None:
            result_lst_cnt = self.result_lst_cnt
        
        woe_list_lst = (
            self.tensor_woe_list_lst
            if self.with_wls_adj==True else
            self.tensor_wls_adj_woe_list_lst
        )
        
        #####################
        # calculate confidence interval
        #####################
        if len(woe_list_lst)>0 and result_lst_cnt>0:
            if self.fit_count%result_lst_cnt==0:
                self.ci_lst = self.ci_lst = tf.transpose(
                    a=np.quantile(
                        a=tf.concat(
                            values=[
                                tf.expand_dims(s0, axis=0)
                                for s0 in woe_list_lst[-result_lst_cnt:]
                            ],
                            axis=0,
                        ).numpy(),
                        q=[self.lower_q, self.upper_q],
                        axis=0,
                        interpolation="linear",
                    ),
                    perm=[1, 2, 0],
                )
                self.lower_ci_lst = self.ci_lst[:, :, 0]
                self.upper_ci_lst = self.ci_lst[:, :, 1]
        elif len(woe_list_lst)>0 and result_lst_cnt==-1:
            self.ci_lst = self.ci_lst = tf.transpose(
                a=np.quantile(
                    a=tf.concat(
                        values=[
                            tf.expand_dims(s0, axis=0)
                            for s0 in woe_list_lst[:]
                        ],
                        axis=0,
                    ).numpy(),
                    q=[self.lower_q, self.upper_q],
                    axis=0,
                    interpolation="linear",
                ),
                perm=[1, 2, 0],
            )
            self.lower_ci_lst = self.ci_lst[:, :, 0]
            self.upper_ci_lst = self.ci_lst[:, :, 1]
    
    ############################################################
    def clear_result_list_lst(
        self,
        ):
        
        self.tensor_woe_list_lst = list()
        self.tensor_wls_adj_woe_list_lst = list()
        self.dict_wls_adj_woe_details_list_lst = list()
        # _ = gc.collect()
    
    ############################################################
    def write_layer_info_data(
        self,
        file=None,
        ):
        
        self._layer_info_data = OrderedDict({
            "in_type": self.in_type,
            "output_with_woe_trans": self.output_with_woe_trans,
            "with_wls_adj": self.with_wls_adj,
            "output_with_wls_adj_details": self.output_with_wls_adj_details,
            "trainable": self.trainable,
            "alpha": self.alpha,
            "lower_q": self.lower_q,
            "upper_q": self.upper_q,
            "result_lst_cnt": self.result_lst_cnt,
            "with_update_result": self.with_update_result,
            "with_calc_confidence_interval": self.with_calc_confidence_interval,
            "fit_count": self.fit_count,
            "fit_samp_count_avg": self.fit_samp_count_avg,
            "lower_ci_lst": (self.lower_ci_lst.numpy() if self.lower_ci_lst is not None else None),
            "upper_ci_lst": (self.upper_ci_lst.numpy() if self.upper_ci_lst is not None else None),
            "ci_lst": (self.ci_lst.numpy() if self.ci_lst is not None else None),
            "_tensor_woe": self._tensor_woe.numpy(),
            "_tensor_wls_adj_woe": self._tensor_wls_adj_woe.numpy(),
            "_dict_wls_adj_woe_details": self._dict_wls_adj_woe_details,
            "tensor_woe_list_lst": [s0.numpy() for s0 in self.tensor_woe_list_lst],
            "tensor_wls_adj_woe_list_lst": [s0.numpy() for s0 in self.tensor_wls_adj_woe_list_lst],
            "dict_wls_adj_woe_details_list_lst": self.dict_wls_adj_woe_details_list_lst,
            "tensor_woe": self.tensor_woe.numpy(),
            "tensor_wls_adj_woe": self.tensor_wls_adj_woe.numpy(),
        })
        
        if file is not None:
            with open(file=file, mode="wb") as fw:
                pickle.dump(self._layer_info_data, file=fw)
        
        self._layer_info_data = None
    
    ############################################################
    def read_layer_info_data(
        self,
        file,
        ):
        
        with open(file=file, mode="rb") as fr:
            self._layer_info_data = pickle.load(file=fr)
        
        self.in_type = self._layer_info_data["in_type"]
        self.output_with_woe_trans = self._layer_info_data["output_with_woe_trans"]
        self.with_wls_adj = self._layer_info_data["with_wls_adj"]
        self.output_with_wls_adj_details = self._layer_info_data["output_with_wls_adj_details"]
        self.trainable = self._layer_info_data["trainable"]
        self.alpha = self._layer_info_data["alpha"]
        self.lower_q = self._layer_info_data["lower_q"]
        self.upper_q = self._layer_info_data["upper_q"]
        self.result_lst_cnt = self._layer_info_data["result_lst_cnt"]
        self.with_update_result = self._layer_info_data["with_update_result"]
        self.with_calc_confidence_interval = self._layer_info_data["with_calc_confidence_interval"]
        self.fit_count = self._layer_info_data["fit_count"]
        self.fit_samp_count_avg = self._layer_info_data["fit_samp_count_avg"]
        self.lower_ci_lst = (tf.constant(self._layer_info_data["lower_ci_lst"]) if self._layer_info_data["lower_ci_lst"] is not None else None)
        self.upper_ci_lst = (tf.constant(self._layer_info_data["upper_ci_lst"]) if self._layer_info_data["upper_ci_lst"] is not None else None)
        self.ci_lst = (tf.constant(self._layer_info_data["ci_lst"]) if self._layer_info_data["ci_lst"] is not None else None)
        self._tensor_woe = tf.constant(self._layer_info_data["_tensor_woe"])
        self._tensor_wls_adj_woe = tf.constant(self._layer_info_data["_tensor_wls_adj_woe"])
        self._dict_wls_adj_woe_details = self._layer_info_data["_dict_wls_adj_woe_details"]
        self.tensor_woe_list_lst = [tf.constant(s0) for s0 in self._layer_info_data["tensor_woe_list_lst"]]
        self.tensor_wls_adj_woe_list_lst = [tf.constant(s0) for s0 in self._layer_info_data["tensor_wls_adj_woe_list_lst"]]
        self.dict_wls_adj_woe_details_list_lst = self._layer_info_data["dict_wls_adj_woe_details_list_lst"]
        self.tensor_woe = tf.constant(self._layer_info_data["tensor_woe"])
        self.tensor_wls_adj_woe = tf.constant(self._layer_info_data["tensor_wls_adj_woe"])
        
        self._layer_info_data = None
    
    
#################################################################





#################################################################
# function
# 分箱（连续型变量）
# @tf.function(autograph=False, experimental_compile=False)
def func_tf_binning_continous(
        tensor_inputs,
        tensor_cut_point,
        out_type="03_onehot_vector",
        right_border=True, include_lowest=False,
    ):
    
    """
    
    ----------------------------------------------------------
    like `func_binning_continuous_v1`
    ----------------------------------------------------------
    """
    
    #####################
    rt = None
    if out_type=="01_info":
        rt = tf.concat(
            values=[
                tf.expand_dims(
                    input=s0,
                    axis=1,
                )
                for s0 in (
                    tf.convert_to_tensor(
                        value=np.array(
                            func_binning_continuous_v1(
                                in_data=tensor_inputs[:, idx],
                                bins=tf.concat(
                                    values=[
                                        [-np.inf],
                                        tf.unique(tensor_cut_point[idx, :]).y,
                                        [np.inf],
                                    ],
                                    axis=0,
                                ),
                                out_type="01_info",
                                right_border=right_border, include_lowest=include_lowest,
                            ).tolist(),
                        ),
                        dtype=tf.string,
                    )
                    for idx in range(tensor_inputs.shape[1])
                )
            ],
            axis=1,
        )
    
    elif out_type=="02_onehot_class":
        _tensor_tmp = func_tf_binning_continous(
            tensor_inputs,
            tensor_cut_point,
            out_type="03_onehot_vector",
            right_border=right_border, include_lowest=include_lowest,
        )
        rt = tf.cast(
            x=tf.sparse.to_dense(
                sp_input=tf.SparseTensor(
                    indices=tf.where(_tensor_tmp==1)[:, :-1],
                    values=tf.where(_tensor_tmp==1)[:, -1],
                    dense_shape=_tensor_tmp.shape[:-1],
                ),
            ),
            dtype=np.uint32,
        )
    
    elif out_type=="03_onehot_vector":
        rt = tf.concat(
            values=[
                tf.expand_dims(
                    input=tf.concat([
                            tf.reshape(s0[:, 0], shape=(-1, 1)),
                            tf.pad(
                                tensor=s0[:, 1:],
                                paddings=[[0, 0],
                                          [(tensor_cut_point.shape[1]+1)-s0[:, 1:].shape[1], 0]],
                                mode="CONSTANT",
                                constant_values=0,
                            ),
                        ],
                        axis=1,
                    ),
                    axis=1,
                )
                for s0 in (
                    tf.convert_to_tensor(
                        value=np.array(
                            func_binning_continuous_v1(
                                in_data=tensor_inputs[:, idx],
                                bins=tf.concat(
                                    values=[
                                        [-np.inf],
                                        tf.unique(tensor_cut_point[idx, :]).y,
                                        [np.inf],
                                    ],
                                    axis=0,
                                ),
                                out_type="03_onehot_vector",
                                right_border=right_border, include_lowest=include_lowest,
                            ).tolist(),
                        ),
                        dtype=tf.int8,
                    )
                    for idx in range(tensor_inputs.shape[1])
                )
            ],
            axis=1,
        )
    else:
        pass
    
    return rt


# #################################################################
# # function
# # 分箱（连续型变量）
# # @tf.function(autograph=False, experimental_compile=False)
# def func_tf_binning_continous(
#         tensor_inputs,
#         tensor_cut_point,
#         out_type="03_onehot_vector",
#         right_border=True, include_lowest=False,
#     ):
    
#     """
    
#     ----------------------------------------------------------
#     like `func_binning_continuous_v1`
#     ----------------------------------------------------------
#     """
    
#     #####################
#     rt = None
#     if out_type=="01_info":
#         rt = tf.concat(
#             values=[
#                 tf.expand_dims(
#                     input=s0,
#                     axis=1,
#                 )
#                 for s0 in (
#                     tf.convert_to_tensor(
#                         value=np.array(
#                             func_binning_continuous_v1(
#                                 in_data=tensor_inputs[:, idx],
#                                 bins=tf.concat(
#                                     values=[
#                                         [-np.inf],
#                                         tf.unique(tensor_cut_point[idx, :]).y,
#                                         [np.inf],
#                                     ],
#                                     axis=0,
#                                 ),
#                                 out_type="01_info",
#                                 right_border=right_border, include_lowest=include_lowest,
#                             ).tolist(),
#                         ),
#                         dtype=tf.string,
#                     )
#                     for idx in range(tensor_inputs.shape[1])
#                 )
#             ],
#             axis=1,
#         )
    
#     elif out_type=="02_onehot_class":
#         _tensor_tmp = func_tf_binning_continous(
#             tensor_inputs,
#             tensor_cut_point,
#             out_type="03_onehot_vector",
#             right_border=right_border, include_lowest=include_lowest,
#         )
#         rt = tf.cast(
#             x=tf.sparse.to_dense(
#                 sp_input=tf.SparseTensor(
#                     indices=tf.where(_tensor_tmp==1)[:, :-1],
#                     values=tf.where(_tensor_tmp==1)[:, -1],
#                     dense_shape=_tensor_tmp.shape[:-1],
#                 ),
#             ),
#             dtype=np.uint32,
#         )
    
#     elif out_type=="03_onehot_vector":
#         rt = tf.concat(
#             values=[
#                 tf.expand_dims(
#                     input=tf.concat([
#                             tf.reshape(s0[:, 0], shape=(-1, 1)),
#                             tf.pad(
#                                 tensor=s0[:, 1:],
#                                 paddings=[[0, 0],
#                                           [(tensor_cut_point.shape[1]+1)-s0[:, 1:].shape[1], 0]],
#                                 mode="CONSTANT",
#                                 constant_values=0,
#                             ),
#                         ],
#                         axis=1,
#                     ),
#                     axis=1,
#                 )
#                 for s0 in (
#                     tf.convert_to_tensor(
#                         value=np.array(
#                             func_binning_continuous_v1(
#                                 in_data=tensor_inputs[:, idx],
#                                 bins=tf.concat(
#                                     values=[
#                                         [-np.inf],
#                                         tf.unique(tensor_cut_point[idx, :]).y,
#                                         [np.inf],
#                                     ],
#                                     axis=0,
#                                 ),
#                                 out_type="03_onehot_vector",
#                                 right_border=right_border, include_lowest=include_lowest,
#                             ).tolist(),
#                         ),
#                         dtype=tf.int8,
#                     )
#                     for idx in range(tensor_inputs.shape[1])
#                 )
#             ],
#             axis=1,
#         )
#     else:
#         pass
    
#     return rt
    
    
#################################################################
# function
# 计算woe
# @tf.function(autograph=False, experimental_compile=False)
def func_tf_calc_woe(
        tensor_X_onehot_vector,
        tensor_y,
        with_wls_adj=False,
        output_with_wls_adj_details=False,
        floating_point=1e-4,
    ):
    
    """
    
    ----------------------------------------------------------
    ref `func_woe_report_v1`
    ----------------------------------------------------------
    """
    
    rt = None
    
    ##################################################
    # 划分好坏样本的特征数据
    _tensor_X_onehot_vector_1_bad = tf.gather(
        params=tensor_X_onehot_vector,
        indices=tf.reshape(tf.where(tensor_y==1), shape=[-1]),
    )
    _tensor_X_onehot_vector_0_good = tf.gather(
        params=tensor_X_onehot_vector,
        indices=tf.reshape(tf.where(tensor_y==0), shape=[-1]),
    )
    
    ##################################################
    # 计算特征 total_pct
    _tensor_total_pct = tf.divide(
        x=tf.reduce_sum(tf.cast(tensor_X_onehot_vector, dtype=tf.float64), axis=0),
        y=tensor_X_onehot_vector.shape[0],
    )
    
    ##################################################
    # 计算特征 WOE
    _tensor_woe = tf.math.log(
        x=tf.divide(
            x=tf.reduce_sum(
                input_tensor=tf.cast(_tensor_X_onehot_vector_1_bad, dtype=tf.float64),
                axis=0,
            ),
            y=_tensor_X_onehot_vector_1_bad.shape[0],
        )+floating_point,
    ) - \
    tf.math.log(
        x=tf.divide(
            x=tf.reduce_sum(
                input_tensor=tf.cast(_tensor_X_onehot_vector_0_good, dtype=tf.float64),
                axis=0,
            ),
            y=_tensor_X_onehot_vector_0_good.shape[0],
        )+floating_point,
    )

    ##################################################
    # 对 woe 进行 wls 回归
    if with_wls_adj==True:
        
        ##################################################
        # 排除 NaN分箱
        _tensor_woe_without_nan = _tensor_woe[:, 1:]
        _tensor_total_pct_without_nan = tf.divide(
            x=_tensor_total_pct[:, 1:],
            y=tf.reshape(tf.reduce_sum(_tensor_total_pct[:, 1:], axis=1), shape=(-1, 1)),
        )
        
        ##################################################
        # 处理wls回归的特征
        _t = tf.broadcast_to(
            input=tf.reshape(
                tf.range(_tensor_total_pct_without_nan.shape[1], dtype=np.int64)+1,
                # tf.range(_tensor_total_pct_without_nan.shape[1])+1,
                shape=(1, -1),
            ),
            shape=_tensor_total_pct_without_nan.shape,
        )
        _t_idx = tf.cumsum(
            x=tf.where(_tensor_total_pct_without_nan!=0, x=1, y=0),
            axis=1,
            reverse=False,
        )
        _tensor_idx = tf.maximum(
            x=_t - tf.reshape(
                tensor=tf.map_fn(
                    fn=lambda s0: tf.reduce_min(tf.where(s0==1)),
                    elems=_t_idx,
                    dtype=_t.dtype,
                ),
                shape=(-1, 1),
            ),
            y=0,
        )
        _tensor_idx = tf.expand_dims(_tensor_idx, axis=2)
        _tensor_idx_p1 = tf.concat([
                tf.cast(tf.where(_tensor_idx==0, x=0, y=1), dtype=_tensor_idx.dtype),
                _tensor_idx,
            ],
            axis=2,
        )
        _tensor_idx_p2 = tf.concat([
                tf.cast(tf.where(_tensor_idx==0, x=0, y=1), dtype=_tensor_idx.dtype),
                _tensor_idx,
                _tensor_idx**2,
            ],
            axis=2,
        )
        
        ##################################################
        # wls求解
        _X_p1 = tf.cast(_tensor_idx_p1, dtype=tf.float64)
        _X_p2 = tf.cast(_tensor_idx_p2, dtype=tf.float64)
        _y = tf.expand_dims(_tensor_woe_without_nan, axis=2)
        _t_W_mask = tf.broadcast_to(
            input=tf.expand_dims(
                input=tf.eye(
                    num_rows=_tensor_total_pct_without_nan.shape[1],
                    num_columns=_tensor_total_pct_without_nan.shape[1],
                    dtype=tf.float64,
                ),
                axis=0,
            ),
            shape=(
                _tensor_total_pct_without_nan.shape[0],
                _tensor_total_pct_without_nan.shape[1],
                _tensor_total_pct_without_nan.shape[1],
            ),
        )
        _t_W = tf.broadcast_to(
            input=tf.expand_dims(_tensor_total_pct_without_nan, axis=2),
            shape=(
                _tensor_total_pct_without_nan.shape[0],
                _tensor_total_pct_without_nan.shape[1],
                _tensor_total_pct_without_nan.shape[1],
            ),
        )
        _W = tf.where(_t_W_mask==1, x=_t_W, y=_t_W_mask)
        
        ##################################################
        # p1
        _tensor_wls_p1_coef = tf.matmul(
            a=tf.matmul(
                a=tf.linalg.pinv(
                    a=tf.matmul(
                        a=tf.matmul(
                            a=_X_p1,
                            b=_W,
                            transpose_a=True,
                            transpose_b=False,
                        ),
                        b=_X_p1,
                        transpose_a=False,
                        transpose_b=False,
                    ),
                ),
                b=tf.matmul(
                    a=_X_p1,
                    b=_W,
                    transpose_a=True,
                    transpose_b=False,
                ),
                transpose_a=False,
                transpose_b=False,
            ),
            b=_y,
            transpose_a=False,
            transpose_b=False,
        )
        _tensor_wls_p1_woe = tf.squeeze(
            input=tf.matmul(
                a=_X_p1,
                b=_tensor_wls_p1_coef,
                transpose_a=False,
                transpose_b=False,
            ),
            axis=-1,
        )
        
        ##################################################
        # p2
        _tensor_wls_p2_coef = tf.matmul(
            a=tf.matmul(
                a=tf.linalg.pinv(
                    a=tf.matmul(
                        a=tf.matmul(
                            a=_X_p2,
                            b=_W,
                            transpose_a=True,
                            transpose_b=False,
                        ),
                        b=_X_p2,
                        transpose_a=False,
                        transpose_b=False,
                    ),
                ),
                b=tf.matmul(
                    a=_X_p2,
                    b=_W,
                    transpose_a=True,
                    transpose_b=False,
                ),
                transpose_a=False,
                transpose_b=False,
            ),
            b=_y,
            transpose_a=False,
            transpose_b=False,
        )
        _tensor_wls_p2_woe = tf.squeeze(
            input=tf.matmul(
                a=_X_p2,
                b=_tensor_wls_p2_coef,
                transpose_a=False,
                transpose_b=False,
            ),
            axis=-1,
        )
        
        ##################################################
        # 计算：R_squared
        _t_sst = tf.reduce_sum(
            input_tensor=((_tensor_woe_without_nan - tf.reshape(
                tensor=tf.reduce_sum(
                    input_tensor=_tensor_woe_without_nan*_tensor_total_pct_without_nan,
                    axis=1,
                ),
                shape=(-1, 1),
            ))**2)*_tensor_total_pct_without_nan,
            axis=1,
        )
        _tensor_rsquared_p1 = 1 - (
            tf.reduce_sum(
                input_tensor=((_tensor_wls_p1_woe-_tensor_woe_without_nan)**2)*_tensor_total_pct_without_nan,
                axis=1,
            )/_t_sst
        )
        _tensor_rsquared_p2 = 1 - (
            tf.reduce_sum(
                input_tensor=((_tensor_wls_p2_woe-_tensor_woe_without_nan)**2)*_tensor_total_pct_without_nan,
                axis=1,
            )/_t_sst
        )
        
        ####################################################
        # 使用一次回归与二次回归的R_squared加权计算woe
        _tensor_wls_adj_woe_without_nan = (
            _tensor_wls_p1_woe*tf.reshape(_tensor_rsquared_p1, shape=(-1, 1))+
            _tensor_wls_p2_woe*tf.reshape(_tensor_rsquared_p2, shape=(-1, 1))
        )/tf.reshape(_tensor_rsquared_p1+_tensor_rsquared_p2, shape=(-1, 1))

        # 使用计算缺失箱的woe填补
        _tensor_wls_adj_woe = tf.concat([
            tf.reshape(_tensor_woe[:, 0], shape=(-1, 1)),
            _tensor_wls_adj_woe_without_nan,
        ], axis=1)
        
        # # 使用0值填补缺失箱的woe
        # _tensor_wls_adj_woe = tf.concat([
        #     tf.zeros(shape=(_tensor_wls_adj_woe_without_nan.shape[0], 1), dtype=tf.float64),
        #     _tensor_wls_adj_woe_without_nan,
        # ], axis=1)
        
        _dict_wls_adj_woe_details = {
            "wls_p1_woe": _tensor_wls_p1_woe.numpy(),
            "wls_p2_woe": _tensor_wls_p2_woe.numpy(),
            "rsquared_p1": _tensor_rsquared_p1.numpy(),
            "rsquared_p2": _tensor_rsquared_p2.numpy(),
            
            "wls_p1_coef": _tensor_wls_p1_coef.numpy(),
            "wls_p2_coef": _tensor_wls_p2_coef.numpy(),
        }
        
        rt = (
            (_tensor_woe, _tensor_wls_adj_woe, _dict_wls_adj_woe_details)
            if output_with_wls_adj_details==True else
            (_tensor_woe, _tensor_wls_adj_woe, None)
        )
    else:
        rt = _tensor_woe, None, None
    
    return rt
    
    
#################################################################




if __name__=="__main__":
    pass

