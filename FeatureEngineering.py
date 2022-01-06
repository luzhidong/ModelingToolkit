# -*- coding: utf-8 -*-
# 
#----------------------------------------------------------
# script name: 
#----------------------------------------------------------
# creator: zhidong.lu
# create date: 2019-05-30
# update date: 2022-01-02
# version: 1.6
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


pd.pandas.set_option("display.max_columns", None)
pd.pandas.set_option("display.max_rows", 300)

plt.style.use({"figure.figsize": [s0*3 for s0 in (4, 2)]})
sns.set(style="whitegrid", rc={"figure.figsize": [s0*3 for s0 in (4, 2)]})



###########################################################################
# 统计dataframe的数据描述
def func_dataframe_describe(in_df, var_names=None, drop_labels=None):
    describe_info = (in_df if var_names==None else in_df[var_names]).drop(labels=([] if drop_labels==None else drop_labels), axis=1) \
        .groupby(axis=1, level=0, sort=False).apply(
        lambda s0: OrderedDict({
            "data_type": ("Numerical" if re.search("(float|int)", s0.dtypes[0].name)!=None else "Categorical"),
            "count": s0.shape[0],
            "count_missing": s0[s0.iloc[:, 0].isna()].shape[0],
            "count_nomissing": s0[-s0.iloc[:, 0].isna()].shape[0],
            "pct_missing": s0[s0.iloc[:, 0].isna()].shape[0]/s0.shape[0],
            "pct_nomissing": s0[-s0.iloc[:, 0].isna()].shape[0]/s0.shape[0],
            "unique_count": s0.iloc[:, 0].unique().shape[0],
            "unique_pct": s0.iloc[:, 0].unique().shape[0]/s0.shape[0],
            
#             "min": (s0.iloc[:, 0].dropna().min() if (re.search("(float|int)", s0.dtypes[0].name)!=None and s0.iloc[:, 0].dropna().shape[0]>0) else np.NaN),
            "min": (s0.iloc[:, 0].dropna().min() if (s0.iloc[:, 0].dropna().shape[0]>0) else np.NaN),
            "mean": (s0.iloc[:, 0].dropna().mean() if (re.search("(float|int)", s0.dtypes[0].name)!=None and s0.iloc[:, 0].dropna().shape[0]>0) else np.NaN),
#             "max": (s0.iloc[:, 0].dropna().max() if (re.search("(float|int)", s0.dtypes[0].name)!=None and s0.iloc[:, 0].dropna().shape[0]>0) else np.NaN),
            "max": (s0.iloc[:, 0].dropna().max() if (s0.iloc[:, 0].dropna().shape[0]>0) else np.NaN),
            
            "percentile_05": (np.percentile(a=s0.iloc[:, 0].dropna(), q=0.05, interpolation="linear") if (re.search("(float|int)", s0.dtypes[0].name)!=None and s0.iloc[:, 0].dropna().shape[0]>0) else np.NaN),
            "percentile_25": (np.percentile(a=s0.iloc[:, 0].dropna(), q=0.25, interpolation="linear") if (re.search("(float|int)", s0.dtypes[0].name)!=None and s0.iloc[:, 0].dropna().shape[0]>0) else np.NaN),
            "percentile_50": (np.percentile(a=s0.iloc[:, 0].dropna(), q=0.50, interpolation="linear") if (re.search("(float|int)", s0.dtypes[0].name)!=None and s0.iloc[:, 0].dropna().shape[0]>0) else np.NaN),
            "percentile_75": (np.percentile(a=s0.iloc[:, 0].dropna(), q=0.75, interpolation="linear") if (re.search("(float|int)", s0.dtypes[0].name)!=None and s0.iloc[:, 0].dropna().shape[0]>0) else np.NaN),
            "percentile_95": (np.percentile(a=s0.iloc[:, 0].dropna(), q=0.95, interpolation="linear") if (re.search("(float|int)", s0.dtypes[0].name)!=None and s0.iloc[:, 0].dropna().shape[0]>0) else np.NaN),

            "top5_value": str(dict(s0.iloc[:, 0].dropna().value_counts().reset_index().values[:5, :])),

            "entropy": np.sum([-t*np.log2(t+1e-2) for t in s0.iloc[:, 0].value_counts()/s0.shape[0]]),
            "entropy_ratio": np.sum([-t*np.log2(t+1e-2) for t in s0.iloc[:, 0].value_counts()/s0.shape[0]])/(np.log(s0.iloc[:, 0].unique().shape[0]+1e-2)+1e-2),
        })
    )
    df_describe_info = pd.DataFrame(describe_info.values.tolist(), index=describe_info.index.values) \
        .reset_index().rename(columns={"index": "column_name"}) \
        .set_index(keys=["column_name"])
    return df_describe_info


###########################################################################
# 统计dataframe对应字段的值频数分布
def func_freqency_stat(in_df, var_names=None, drop_labels=None):
    _tmp = [[
        _col,
        _df.dtypes[0].name,
        _df.shape[0],
        _df.iloc[:, 0].value_counts(dropna=False),
    ] for _col, _df in (in_df if var_names==None else in_df[var_names]).drop(labels=([] if drop_labels==None else drop_labels), axis=1) \
            .groupby(axis=1, level=0, sort=False)
    ]
    df_freq_table = pd.concat([
        pd.DataFrame([OrderedDict({
            "column_name": _col,
            "data_type": ("Numerical" if re.search("(float|int)", _dtype_name)!=None else "Categorical"),
            "value": ("NaN" if pd.isna(_value) else _value),
            "count": _count,
            "count_cum": _count_cum,
            "pct": _count/_df_size,
            "pct_cum": _count_cum/_df_size,
        }) for _value, _count, _count_cum in zip(_data.index.tolist(), _data.values, _data.cumsum())])
        for _col, _dtype_name, _df_size, _data in _tmp
    ], ignore_index=True)
    
    return df_freq_table


###########################################################################
# 计算关联表匹配率（主键匹配率、外键匹配率）
def func_table_match_rate(df_primary, var_foreign,
                          df_foreign, var_f_primary):
    ######################################################
    _tmp = df_primary[var_foreign].merge(
        right=df_foreign[var_f_primary].reset_index(),
        how="left", left_on=var_foreign, right_on=var_f_primary,
    )
    primary_match_rate = _tmp[_tmp["index"].notna()].shape[0]/_tmp.shape[0]
    
    ######################################################
    _tmp = df_foreign[var_f_primary].merge(
        right=df_primary[var_foreign].drop_duplicates().reset_index(),
        how="left", left_on=var_f_primary, right_on=var_foreign,
    )
    foreign_match_rate = _tmp[_tmp["index"].notna()].shape[0]/_tmp.shape[0]
    
    rt = {
        "primary_match_rate": primary_match_rate,
        "foreign_match_rate": foreign_match_rate,
    }
    return rt


###########################################################################
# 分箱（连续型变量_分位数）
# 优先输出上分位点
def func_binning_continuous_quantile_v1(
        in_data, bins_q,
        out_type="01_info",
        right_border=True, include_lowest=False,
    ):
    
    x = pd.Series(in_data)
    bins_q = [s0 for s0 in bins_q if s0>0 and s0<100]
    bins_q_cp = np.percentile(
        a=x.dropna(),
        q=bins_q,
        interpolation="linear",
    ).tolist()
    
    ################################################
    _bins_q = [0]+bins_q+[100]
    _bins_q_cp = [-np.inf]+bins_q_cp+[np.inf]
    _qcut_mapping = pd.Series(dict(zip(_bins_q, _bins_q_cp,)))
    
    ################################################
    _df_qcut_mapping = _qcut_mapping \
        .reset_index(name="cp") \
        .rename(columns={"index": "q"}) \
        .reset_index() \
        .rename(columns={"index": "idx_output"})
    _df_qcut_mapping["q_lag"] = _df_qcut_mapping["q"].shift(periods=1)
    _df_qcut_mapping["cp_lag"] = _df_qcut_mapping["cp"].shift(periods=1)
    _df_qcut_mapping = _df_qcut_mapping \
        [["idx_output", "q_lag", "q", "cp_lag", "cp"]] \
        .reset_index(drop=True)

    _df_qcut_mapping_dups = _df_qcut_mapping \
        .drop_duplicates(subset=["cp"], keep="last") \
        .reset_index(drop=True) \
        .reset_index(drop=False) \
        .rename(columns={"index": "idx_dups"}) \
        .set_index("idx_dups")
    
    if out_type=="01_info":
        rt = pd.cut(
            x=x,
            bins=_df_qcut_mapping_dups["cp"].tolist(),
            right=right_border, include_lowest=include_lowest,
            labels=[s0 for s0 in range(_df_qcut_mapping_dups.shape[0]-1)],
        ).astype(float).apply(
            lambda s0: (
                "{:0{}d}_{}NaN, NaN{}".format(0, int(np.log10(_df_qcut_mapping_dups["idx_output"].max()))+1, ("(" if right_border else "["), ("]" if right_border else ")"))
                if pd.isna(s0) else
                "{:0{}d}_{}{:.2f}%, {:.2f}%{}".format(
                    # int(s0)+1,
                    _df_qcut_mapping_dups["idx_output"].get(int(s0)+1),
                    int(np.log10(_df_qcut_mapping_dups["idx_output"].max()))+1,
                    ("(" if right_border and s0!=0 else "["),
                    # _df_qcut_mapping_dups["q_lag"].get(int(s0+1)),
                    _df_qcut_mapping_dups["q"].get(int(s0)),
                    _df_qcut_mapping_dups["q"].get(int(s0+1)),
                    ("]" if right_border else ")"),
                )
            )
        )
    elif out_type=="02_onehot_class":
        rt = pd.cut(
            x=x,
            bins=_df_qcut_mapping_dups["cp"].tolist(),
            right=right_border, include_lowest=include_lowest,
            labels=[s0 for s0 in range(_df_qcut_mapping_dups.shape[0]-1)],
        ).astype(float).apply(
            lambda s0: (
                0 if pd.isna(s0) else
                _df_qcut_mapping_dups["idx_output"].get(int(s0)+1)
            )
        ).astype(np.int8)
    elif out_type=="03_onehot_vector":
        rt = pd.cut(
            x=x,
            bins=_df_qcut_mapping_dups["cp"].tolist(),
            right=right_border, include_lowest=include_lowest,
            labels=[s0 for s0 in range(_df_qcut_mapping_dups.shape[0]-1)],
        ).astype(float).apply(
            lambda s0: (
                0 if pd.isna(s0) else
                _df_qcut_mapping_dups["idx_output"].get(int(s0)+1)
            )
        ).astype(np.int8)
        
        _num_class = _df_qcut_mapping_dups["idx_output"].max()+1
        _arr = rt.values
        rt = pd.Series(
            data=np.eye(N=_num_class, dtype=np.int8)[_arr].tolist(),
        ).apply(lambda s0: np.array(s0))
    else:
        rt = None
    return rt


###########################################################################
# 分箱（连续型变量）
def func_binning_continuous_v1(
        in_data, bins,
        out_type="01_info",
        right_border=True, include_lowest=False,
    ):
    x = pd.Series(in_data)
    bins_cnt = len(bins)-1
    
    if out_type=="01_info":
        rt = pd.cut(
            x=x,
            bins=bins, right=right_border, include_lowest=include_lowest,
            labels=[s0 for s0 in range(bins_cnt)], 
        ).astype(float).apply(
            lambda s0: (
                "{:0{}d}_{}NaN, NaN{}".format(0, int(np.log10(bins_cnt+1))+1, ("(" if right_border else "["), ("]" if right_border else ")"))
                if pd.isna(s0) else
                "{:0{}d}_{}{:.4f}, {:.4f}{}".format(int(s0)+1, int(np.log10(bins_cnt+1))+1, ("(" if right_border else "["), bins[int(s0)], bins[int(s0)+1], ("]" if right_border else ")"))
            )
        )
    elif out_type=="02_onehot_class":
        rt = pd.cut(
            x=x,
            bins=bins, right=right_border, include_lowest=include_lowest,
            labels=[s0 for s0 in range(bins_cnt)], 
        ).astype(float).apply(
            lambda s0: (
                0 if pd.isna(s0) else
                int(s0)+1
            )
        ).astype(np.int8)
    elif out_type=="03_onehot_vector":
        rt = pd.cut(
            x=x,
            bins=bins, right=right_border, include_lowest=include_lowest,
            labels=[s0 for s0 in range(bins_cnt)], 
        ).astype(float).apply(
            lambda s0: (
                0 if pd.isna(s0) else
                int(s0)+1
            )
        ).astype(np.int8)
        
        _num_class = bins_cnt+1
        _arr = rt.values
        rt = pd.Series(
            data=np.eye(N=_num_class, dtype=np.int8)[_arr].tolist(),
        ).apply(lambda s0: np.array(s0))
    else:
        rt = None
    return rt

def func_binning_continuous_v2(
        in_df, var_name, bins,
        out_type="01_info",
        right_border=True, include_lowest=False,
    ):
    rt = func_binning_continuous_v1(
        in_data=in_df[var_name], bins=bins, 
        out_type=out_type,
        right_border=right_border, include_lowest=include_lowest,
    )
    return rt


###########################################################################
# 合箱（离散型变量）
def func_combining_discrete_v1(in_data, mapping_gb_class, fillna_value="NaN", cvt_fillna_value=0):
    mapping_gb_class[fillna_value] = cvt_fillna_value
    _mapping_gb_class_keys = set(mapping_gb_class.keys())
    in_data = pd.Series([(s0 if s0 in _mapping_gb_class_keys else "NaN") for s0 in pd.Series(in_data).fillna("NaN")])
    rt = in_data.apply(lambda s0: mapping_gb_class.get(s0))
    return rt

def func_combining_discrete_v2(in_df, var_name, fillna_value="NaN", cvt_fillna_value=0):
    rt = func_combining_discrete_v1(
        in_data=in_df[var_name], mapping_gb_class=mapping_gb_class,
        fillna_value=fillna_value, cvt_fillna_value=cvt_fillna_value,
    )
    return rt


###########################################################################
# 自动分箱（连续型变量）
def func_auto_binning_continuous_v1(
        in_var, in_target, min_pct=0.05, max_bins_cnt=None, right_border=True, include_lowest=False, method="02_decision_tree",
        with_lift_ks=False, lift_calc_ascending=True,
        with_wls_adj_woe=False,
    ):
    max_bins_cnt = (int(1/min_pct)+1 if max_bins_cnt==None else max_bins_cnt)
    data = in_var
    data_notna = pd.Series([s0 for s0 in in_var if not pd.isna(s0)])
    in_target_notna = pd.Series([s1 for s0, s1 in zip(in_var, in_target) if not pd.isna(s0)])
    
    #####################################################################################
    if method=="01_equal_freq":
        
        if len(data_notna)>0:
            bins_cnt = min(max_bins_cnt, (int(1/min_pct) if min_pct else max_bins_cnt))

            q = [(s0+1)/bins_cnt*100 for s0 in range(bins_cnt-1)]
            boundary = np.percentile(data_notna, q=q, interpolation="linear")
            
            boundary = np.unique(boundary).tolist()
            boundary.sort()
            boundary = [-np.inf]+boundary+[np.inf]
        else:
            boundary = [-np.inf, np.inf]
        
    elif method=="02_decision_tree":
        
        if len(data_notna)>0:
            from sklearn.tree import DecisionTreeClassifier
            clf = DecisionTreeClassifier(
                # criterion="entropy",
                criterion="gini",
                splitter="best",
                max_leaf_nodes=max_bins_cnt,
                min_samples_leaf=min_pct,
                min_samples_split=2*min_pct+0.0001,
                # min_impurity_decrease=0.000000001,
            )
            clf.fit(X=np.array(data_notna).reshape(-1, 1), y=in_target_notna)

            boundary = [clf.tree_.threshold[idx] for idx in range(clf.tree_.node_count) if clf.tree_.children_left[idx]!=clf.tree_.children_right[idx]]
            boundary.sort()
            boundary = [-np.inf]+boundary+[np.inf]
        else:
            boundary = [-np.inf, np.inf]
    
    #####################################################################################
    elif method=="03_best_ks":
        data_size = data.shape[0]
        _target_labels = in_target.unique().tolist()
        
        crosstab = pd.crosstab(index=data, columns=in_target)
        crosstab.index.name = "index"
        crosstab.columns.name = ""
        crosstab = crosstab.reset_index().rename(columns={"index": "value"})

        def _func_ks_stat(in_crosstab):
            if in_crosstab.shape[0]>0:
                crosstab = in_crosstab.reset_index(drop=True)
                crosstab[["{}_%".format(s0) for s0 in _target_labels]] = (crosstab[_target_labels]/crosstab[_target_labels].sum()).fillna(0)
                crosstab[["{}_%_cum".format(s0) for s0 in _target_labels]] = crosstab[["{}_%".format(s0) for s0 in _target_labels]].cumsum()
                crosstab["diff_abs"] = crosstab[["{}_%_cum".format(s0) for s0 in _target_labels]].apply(lambda s0: abs(s0[0]-s0[1]), axis=1)
                ks_value = crosstab["diff_abs"].max()
                return crosstab, ks_value
            else:
                return None, None

        def _func_cut_ks_value(in_crosstab, ks_value):
            crosstab = in_crosstab.reset_index(drop=True)
            cut_stats = crosstab[crosstab["diff_abs"]==ks_value]
            cut_value = cut_stats["value"].values[0]
            crosstab_left_eq, ks_value_left_eq = _func_ks_stat(in_crosstab=crosstab.query("value<={}".format(cut_value))[["value"]+_target_labels])
            crosstab_right, ks_value_right = _func_ks_stat(crosstab.query("value>{}".format(cut_value))[["value"]+_target_labels])
            return cut_value, crosstab_left_eq, crosstab_right, ks_value_left_eq, ks_value_right

        crosstab, ks_value = _func_ks_stat(in_crosstab=crosstab)
        cut_value, _, _, _, _ = _func_cut_ks_value(in_crosstab=crosstab, ks_value=ks_value)
        _crosstab_pending = [{"ks_value": ks_value, "crosstab": crosstab, "size": crosstab[_target_labels].sum().sum()}]
        boundary = [cut_value]
        while 1:
            _max_ks = max([s0["ks_value"] for s0 in _crosstab_pending])
            _t1 = [s0 for s0 in _crosstab_pending if s0["ks_value"]==_max_ks]
            _t2 = [s0 for s0 in _crosstab_pending if s0["ks_value"]!=_max_ks]
#             _max_size 0= max([s0["size"] for s0 in _crosstab_pending])
#             _t1 = [s0 for s0 in _crosstab_pending if s0["size"]==_max_size]
#             _t2 = [s0 for s0 in _crosstab_pending if s0["size"]!=_max_size]
            _crosstab = _t1[0]["crosstab"]
            _crosstab_pending = _t1[1:]+_t2

            if _crosstab[_target_labels].sum().sum()/data_size>=min_pct and _crosstab.shape[0]>1:
                _crosstab, ks_value = _func_ks_stat(in_crosstab=_crosstab)
                cut_value, crosstab_left_eq, crosstab_right, ks_value_left_eq, ks_value_right = _func_cut_ks_value(in_crosstab=_crosstab, ks_value=ks_value)

                if ((ks_value_left_eq!=None and ks_value_right!=None) and \
                    (crosstab_left_eq[_target_labels].sum().sum()/data_size>=min_pct and crosstab_right[_target_labels].sum().sum()/data_size>=min_pct)):
                    boundary.append(cut_value)
                    boundary = list(set(boundary))
                    _crosstab_pending = _crosstab_pending+[{"ks_value": ks_value_left_eq, "crosstab": crosstab_left_eq, "size": crosstab_left_eq[_target_labels].sum().sum()},
                                                           {"ks_value": ks_value_right, "crosstab": crosstab_right, "size": crosstab_right[_target_labels].sum().sum()}]

            if len(boundary)+1>=max_bins_cnt or len(_crosstab_pending)==0:
                break
        
        boundary = np.unique(boundary).tolist()
        boundary.sort()
        boundary = [-np.inf]+boundary+[np.inf]
    #####################################################################################
    
    data_converted = func_binning_continuous_v1(
        in_data=in_var, bins=boundary,
        out_type="01_info",
        right_border=right_border, include_lowest=include_lowest,
    )
    crosstab_converted = func_woe_report_v1(
        in_var=data_converted, in_target=in_target, with_total=True,
        with_lift_ks=with_lift_ks, lift_calc_ascending=lift_calc_ascending,
        with_wls_adj_woe=with_wls_adj_woe,
    )
    
    return data_converted, crosstab_converted, boundary

def func_auto_binning_continuous_v2(
        in_df, var_name, target_label, min_pct=0.05, max_bins_cnt=None, right_border=True, include_lowest=False, method="02_decision_tree",
        with_lift_ks=False, lift_calc_ascending=True,
        with_wls_adj_woe=False,
    ):
    data_converted, crosstab_converted, boundary = \
        func_auto_binning_continuous_v1(
            in_var=in_df[var_name], in_target=in_df[target_label],
            min_pct=min_pct, max_bins_cnt=max_bins_cnt, right_border=right_border, include_lowest=include_lowest, method=method,
            with_lift_ks=with_lift_ks, lift_calc_ascending=lift_calc_ascending,
            with_wls_adj_woe=with_wls_adj_woe,
        )
    return data_converted, crosstab_converted, boundary


###########################################################################
# 自动合箱（离散型变量）
def func_auto_combining_discrete_v1(
        in_var, in_target, min_pct=0.05, max_bins_cnt=None, method="01_equal_freq",
        with_lift_ks=False, lift_calc_ascending=True,
        with_wls_adj_woe=False,
    ):
    max_bins_cnt = (np.ceil(1/min_pct) if max_bins_cnt==None else max_bins_cnt)
    max_pct = (min_pct if 1/max_bins_cnt<min_pct else 1/max_bins_cnt)
    data = pd.Series(in_var).fillna("NaN")
    data_size = data.shape[0]
    _target_labels = in_target.unique().tolist()
    
    crosstab = func_woe_report_v1(
        in_var=data, in_target=in_target, with_total=False,
        with_lift_ks=with_lift_ks, lift_calc_ascending=lift_calc_ascending,
        with_wls_adj_woe=with_wls_adj_woe,
    )
    crosstab = crosstab.rename(columns=dict([("{}_#".format(t), t) for t in _target_labels]))
    crosstab = crosstab.sort_values(by=["bad_rate", "total_pct"], ascending=[False, False])
    crosstab.index.name = "index"
    crosstab = crosstab.reset_index().rename(columns={"index": "value"}).reset_index().rename(columns={"index": "idx"})
    
    #####################################################################################
    if method=="01_equal_freq":
        gb_class = 1
        total_pct_cum = 0
        total_pct_remain = 1
        _mapping_data = []
        for idx, good_cnt, bad_cnt, total_pct in crosstab[["value"]+_target_labels+["total_pct"]].values[:]:
            if good_cnt==0 or bad_cnt==0:
                total_pct_cum = total_pct_cum+total_pct
            else:
                if total_pct_cum<=max_pct:
                    total_pct_cum = total_pct_cum+total_pct
                else:
                    if gb_class+1<=max_bins_cnt:
                        if total_pct_remain>=min_pct:
                            gb_class = gb_class+1
                        total_pct_cum = total_pct
                    else:
                        total_pct_cum = total_pct_cum+total_pct
                        gb_class = max_bins_cnt

            total_pct_remain = total_pct_remain-total_pct
            _mapping_data.append([idx, gb_class])
            # print(idx, gb_class, total_pct, total_pct_cum, total_pct_remain)
    
    #####################################################################################
    elif method=="02_best_ks":
        def _func_ks_stat(in_crosstab):
            if in_crosstab.shape[0]>0:
                crosstab = in_crosstab.reset_index(drop=True)
                crosstab[["{}_%".format(s0) for s0 in _target_labels]] = (crosstab[_target_labels]/crosstab[_target_labels].sum()).fillna(0)
                crosstab[["{}_%_cum".format(s0) for s0 in _target_labels]] = crosstab[["{}_%".format(s0) for s0 in _target_labels]].cumsum()
                crosstab["diff_abs"] = crosstab[["{}_%_cum".format(s0) for s0 in _target_labels]].apply(lambda s0: abs(s0[0]-s0[1]), axis=1)
                ks_value = crosstab["diff_abs"].max()
                return crosstab, ks_value
            else:
                return None, None

        def _func_cut_ks_value(in_crosstab, ks_value):
            crosstab = in_crosstab.reset_index(drop=True)
            cut_stats = crosstab[crosstab["diff_abs"]==ks_value]
            cut_idx = cut_stats["idx"].values[0]
            crosstab_left_eq, ks_value_left_eq = _func_ks_stat(in_crosstab=crosstab.query("idx<={}".format(cut_idx))[["idx", "value"]+_target_labels])
            crosstab_right, ks_value_right = _func_ks_stat(crosstab.query("idx>{}".format(cut_idx))[["idx", "value"]+_target_labels])
            return cut_idx, crosstab_left_eq, crosstab_right, ks_value_left_eq, ks_value_right

        crosstab, ks_value = _func_ks_stat(in_crosstab=crosstab)
        cut_idx, _, _, _, _ = _func_cut_ks_value(in_crosstab=crosstab, ks_value=ks_value)
        _crosstab_pending = [{"ks_value": ks_value, "crosstab": crosstab, "size": crosstab[_target_labels].sum().sum()}]
        boundary_idx = [cut_idx]
        while 1:
            _max_ks = max([s0["ks_value"] for s0 in _crosstab_pending])
            _t1 = [s0 for s0 in _crosstab_pending if s0["ks_value"]==_max_ks]
            _t2 = [s0 for s0 in _crosstab_pending if s0["ks_value"]!=_max_ks]
#             _max_size = max([s0["size"] for s0 in _crosstab_pending])
#             _t1 = [s0 for s0 in _crosstab_pending if s0["size"]==_max_size]
#             _t2 = [s0 for s0 in _crosstab_pending if s0["size"]!=_max_size]
            _crosstab = _t1[0]["crosstab"]
            _crosstab_pending = _t1[1:]+_t2

            if _crosstab[_target_labels].sum().sum()/data_size>=min_pct and _crosstab.shape[0]>1:
                _crosstab, ks_value = _func_ks_stat(in_crosstab=_crosstab)
                cut_idx, crosstab_left_eq, crosstab_right, ks_value_left_eq, ks_value_right = _func_cut_ks_value(in_crosstab=_crosstab, ks_value=ks_value)

                if ((ks_value_left_eq!=None and ks_value_right!=None) and \
                    (crosstab_left_eq[_target_labels].sum().sum()/data_size>=min_pct and crosstab_right[_target_labels].sum().sum()/data_size>=min_pct)):
                    boundary_idx.append(cut_idx)
                    boundary_idx = list(set(boundary_idx))
                    _crosstab_pending = _crosstab_pending+[{"ks_value": ks_value_left_eq, "crosstab": crosstab_left_eq, "size": crosstab_left_eq[_target_labels].sum().sum()},
                                                           {"ks_value": ks_value_right, "crosstab": crosstab_right, "size": crosstab_right[_target_labels].sum().sum()}]

            if len(boundary_idx)+1>=max_bins_cnt or len(_crosstab_pending)==0:
                break
        boundary_idx.sort()

        crosstab["retain_flag"] = crosstab["idx"].apply(lambda s0: s0 in boundary_idx)
        _mapping_data = []
        gb = 1
        for _value, _retain_flag in zip(crosstab["value"], crosstab["retain_flag"]):
            _mapping_data.append([_value, gb])
            if _retain_flag==True:
                gb = gb+1
    
    #####################################################################################
    mapping_gb_class = dict(_mapping_data)
    data_converted = data.apply(lambda s0: mapping_gb_class.get(s0))
    crosstab_converted = func_woe_report_v1(
        in_var=data_converted, in_target=in_target, with_total=True,
        with_lift_ks=with_lift_ks, lift_calc_ascending=lift_calc_ascending,
        with_wls_adj_woe=with_wls_adj_woe,
    )
    return data_converted, crosstab_converted, mapping_gb_class

def func_auto_combining_discrete_v2(
        in_df, var_name, target_label, min_pct=0.05, max_bins_cnt=None, method="01_equal_freq",
        with_lift_ks=False, lift_calc_ascending=True,
        with_wls_adj_woe=False,
    ):
    data_converted, crosstab_converted, mapping_gb_class = \
        func_auto_combining_discrete_v1(
            in_var=in_df[var_name], in_target=in_df[target_label],
            min_pct=min_pct, max_bins_cnt=max_bins_cnt, method=method,
            with_lift_ks=with_lift_ks, lift_calc_ascending=lift_calc_ascending,
            with_wls_adj_woe=with_wls_adj_woe,
        )
    return data_converted, crosstab_converted, mapping_gb_class


###########################################################################
# 分层过采样
def func_oversample_stratify(in_df, n, stratify_key, group_weight, random_seed):
    _df = in_df.reset_index(drop=True)
    group_weight = dict(pd.Series(group_weight)/sum(pd.Series(group_weight)))
    rt = pd.DataFrame([])
    for _idx, (_v, _pct) in list(enumerate(group_weight.items()))[:]:
        if _idx<len(group_weight.items())-1:
            _cnt = int(n*_pct)
        else:
            _cnt = n-rt.shape[0]
        _df_g = _df[_df[stratify_key]==_v]
        _cnt_g = _df_g.shape[0]
        if _cnt<=_cnt_g:
            rt = rt.append(_df_g, ignore_index=True)
        else:
            rt = rt.append(
                pd.concat([
                    _df_g,
                    _df_g.sample(n=_cnt-_cnt_g, replace=True, random_state=random_seed)
                ], ignore_index=True),
                ignore_index=True,
            )
    return rt

###########################################################################
# 计算变量WOE报告
def func_woe_report_v1(
        in_var,
        in_target,
        with_total=True, good_label_val="0_good", bad_label_val="1_bad",
        floating_point=1e-4,
        with_lift_ks=False, lift_calc_ascending=True,
        with_wls_adj_woe=False,
    ):
    in_var = pd.Series(in_var).fillna(value="NaN")
    
    crosstab = pd.crosstab(index=in_var, columns=in_target)
    _rename = columns=dict(zip(crosstab.columns.tolist(), ["{}_#".format(s0) for s0 in crosstab.columns.tolist()]))
    crosstab[["{}_%".format(s0) for s0 in crosstab.columns.tolist()]] = crosstab[crosstab.columns.tolist()]/crosstab.sum(axis=0)
    crosstab = crosstab.rename(columns=_rename)

    crosstab["WOE"] = crosstab[["{}_%".format(good_label_val), "{}_%".format(bad_label_val)]].apply(lambda s0: np.log(s0[1]+floating_point)-np.log(s0[0]+floating_point), axis=1)
    crosstab["IV"] = crosstab[["{}_%".format(good_label_val), "{}_%".format(bad_label_val)]+["WOE"]].apply(lambda s0: (s0[1]-s0[0])*s0["WOE"], axis=1)
    crosstab["total"] = crosstab[[s0 for s0 in crosstab.columns.tolist() if ("#" in s0)]].apply(lambda s0: s0.sum(), axis=1).astype(int)
    crosstab["total_pct"] = crosstab[[s0 for s0 in crosstab.columns.tolist() if ("#" in s0)]].sum(axis=1)/crosstab[[s0 for s0 in crosstab.columns.tolist() if ("#" in s0)]].values.sum()
    crosstab.index.name = "index"
    crosstab.columns.name = ""
    
    if with_wls_adj_woe:
        _data_wls = crosstab[["WOE", "total_pct"]] \
            .reset_index() \
            .rename(columns={"index": "_index"})
        _data_wls = _data_wls[_data_wls["_index"].apply(lambda s0: int(s0.split("_")[0])!=0)]
        _data_wls = _data_wls \
            .reset_index(drop=False) \
            .rename(columns={"index": "idx"})
        _data_wls["idx_2"] = _data_wls["idx"]**2
        _data_wls["Intercept"] = 1
        _data_wls = _data_wls[[
            "_index", "total_pct", "WOE", "Intercept", "idx", "idx_2",
        ]].reset_index(drop=True)
        
        _cols_wls_p1 = ["Intercept", "idx"]
        _cols_wls_p2 = ["Intercept", "idx", "idx_2"]
        if _data_wls.shape[0]==1:
            _data_wls[["idx"]] = 0
            _data_wls[["idx", "idx_2"]] = 0
        elif _data_wls.shape[0]==2:
            _data_wls[["idx_2"]] = 0
        _X_p1 = _data_wls[_cols_wls_p1]
        _X_p2 = _data_wls[_cols_wls_p2]
        _y = _data_wls["WOE"]
        _weights = _data_wls["total_pct"]/_data_wls["total_pct"].sum()
        _W = (np.eye(_weights.shape[0], _weights.shape[0])*_weights.values)
        
        ###########
        # wls求解
        ###########
        
        # ####################################################
        # _model_wls = sm.WLS(
        #     endog=_y,
        #     exog=_X_p1,
        #     hasconst=True,
        #     weights=_weights,
        # )
        # _model_wls_res = _model_wls.fit(method="pinv")
        # _data_wls["_{}_coef".format("wls_p1")] = _data_wls.apply(lambda s0: _model_wls_res.params.tolist(), axis=1)
        # _data_wls["{}_WOE".format("wls_p1")] = _model_wls_res.predict(exog=_X_p1)
        
        # ####################################################
        # _model_wls = sm.WLS(
        #     endog=_y,
        #     exog=_X_p2,
        #     hasconst=True,
        #     weights=_weights,
        # )
        # _model_wls_res = _model_wls.fit(method="pinv")
        # _data_wls["_{}_coef".format("wls_p2")] = _data_wls.apply(lambda s0: _model_wls_res.params.tolist(), axis=1)
        # _data_wls["{}_WOE".format("wls_p2")] = _model_wls_res.predict(exog=_X_p2)
        
        _data_wls["_{}_coef".format("wls_p1")] = _data_wls.apply(
            lambda s0: np.linalg.pinv(_X_p1.T.dot(_W).dot(_X_p1)) \
                        .dot(_X_p1.T.dot(_W)) \
                        .dot(_y) \
                        .tolist(),
            axis=1,
        )
        _data_wls["_{}_coef".format("wls_p2")] = _data_wls.apply(
            lambda s0: np.linalg.pinv(_X_p2.T.dot(_W).dot(_X_p2)) \
                        .dot(_X_p2.T.dot(_W)) \
                        .dot(_y) \
                        .tolist(),
            axis=1,
        )
        _data_wls["{}_WOE".format("wls_p1")] = _data_wls.apply(
            lambda s0: s0["Intercept"]*s0["_wls_p1_coef"][0] + \
                       s0["idx"]*s0["_wls_p1_coef"][1],
            axis=1,
        )
        _data_wls["{}_WOE".format("wls_p2")] = _data_wls.apply(
            lambda s0: s0["Intercept"]*s0["_wls_p2_coef"][0] + \
                       s0["idx"]*s0["_wls_p2_coef"][1] + \
                       s0["idx_2"]*s0["_wls_p2_coef"][2],
            axis=1,
        )
        
        ####################################################
        # 计算：R_squared
        _rsquared_p1 = 1-(((_data_wls["wls_p1_WOE"]-_data_wls["WOE"])**2)*_weights).sum()/(((_data_wls["WOE"]-(_data_wls["WOE"]*_weights).sum())**2)*_weights).sum()
        # _rsquared_adj_p1 = (
        #     np.NaN
        #     if (_data_wls.shape[0]-len([s0 for s0 in _cols_wls_p1 if s0!="Intercept"])-1)==0 else
        #     1-(1-_rsquared_p1)*((_data_wls.shape[0]-1)/(_data_wls.shape[0]-len([s0 for s0 in _cols_wls_p1 if s0!="Intercept"])-1))
        # )
        # print(_rsquared_p1, _rsquared_adj_p1)
        
        _rsquared_p2 = 1-(((_data_wls["wls_p2_WOE"]-_data_wls["WOE"])**2)*_weights).sum()/(((_data_wls["WOE"]-(_data_wls["WOE"]*_weights).sum())**2)*_weights).sum()
        # _rsquared_adj_p2 = (
        #     np.NaN
        #     if (_data_wls.shape[0]-len([s0 for s0 in _cols_wls_p2 if s0!="Intercept"])-1)==0 else
        #     1-(1-_rsquared_p2)*((_data_wls.shape[0]-1)/(_data_wls.shape[0]-len([s0 for s0 in _cols_wls_p2 if s0!="Intercept"])-1))
        # )
        # print(_rsquared_p2, _rsquared_adj_p2)
        
        # # print(_model_wls_res.rsquared, _model_wls_res.rsquared_adj)
        
        ####################################################
        # _t = _model_wls_res.summary2().tables
        # _df_model_wls_summary = _t[0]
        # _df_model_wls_params = _t[1]
        # _model_wls_res.summary2()
        
        ####################################################
        # 使用一次回归与二次回归的R_squared加权计算woe
        _data_wls["wls_adj_WOE"] = _data_wls.apply(
            lambda s0: (_rsquared_p1*s0["wls_p1_WOE"]+_rsquared_p2*s0["wls_p2_WOE"])/(_rsquared_p1+_rsquared_p2),
            axis=1,
        )
        
        ####################################################
        # 结果加入到 crosstab 中
        _mapping = dict(_data_wls[["_index", "wls_adj_WOE"]].values)
        crosstab["wls_adj_WOE"] = [_mapping.get(s0) for s0 in crosstab.index]
        # crosstab.loc[crosstab["wls_adj_WOE"].isna(), "wls_adj_WOE"] = \
        #     crosstab.loc[crosstab["wls_adj_WOE"].isna(), "WOE"]
        # crosstab.loc[crosstab["wls_adj_WOE"].isna(), "wls_adj_WOE"] = 0
        crosstab.loc[crosstab["wls_adj_WOE"].isna(), "wls_adj_WOE"] = np.NaN
        
        _mapping = _data_wls.set_index(keys="_index").apply(
                lambda s0: {
                    "wls_p1_WOE": s0["wls_p1_WOE"],
                    "wls_p2_WOE": s0["wls_p2_WOE"],
                    "_rsquared_p1": _rsquared_p1,
                    "_rsquared_p2": _rsquared_p2,
                },
                axis=1,
            ).to_dict()
        crosstab["_wls_adj_woe_details"] = [_mapping.get(s0, np.NaN) for s0 in crosstab.index]
        
    if with_lift_ks:
        crosstab = crosstab.sort_index(ascending=lift_calc_ascending)
        
        crosstab["cum_1_bad_%"] = crosstab["1_bad_%"].cumsum()
        crosstab["cum_0_good_%"] = crosstab["0_good_%"].cumsum()
        crosstab["cum_total_pct"] = crosstab["total_pct"].cumsum()
        crosstab["lift"] = crosstab["cum_1_bad_%"]/crosstab["cum_total_pct"]
        crosstab["ks"] = crosstab.apply(lambda s0: abs(s0["cum_0_good_%"]-s0["cum_1_bad_%"]), axis=1)
        
        crosstab = crosstab.sort_index(ascending=True)

    if with_total:
        _total = pd.DataFrame(crosstab.sum(axis=0), columns=["total"]).T
        _total["WOE"] = np.NaN
        if with_lift_ks:
            _total["cum_1_bad_%"] = np.NaN
            _total["cum_0_good_%"] = np.NaN
            _total["cum_total_pct"] = np.NaN
            _total["lift"] = np.NaN
            _total["ks"] = crosstab["ks"].max()
        crosstab = crosstab.append(_total)
        crosstab[[s0 for s0 in crosstab.columns.tolist() if ("#" in s0)]+["total"]] = crosstab[[s0 for s0 in crosstab.columns.tolist() if ("#" in s0)]+["total"]].astype(int)
    
    crosstab["bad_rate"] = crosstab.iloc[:, 1]/crosstab[[s0 for s0 in crosstab.columns.tolist() if ("#" in s0)]].sum(axis=1)
    
    crosstab = crosstab[
        [
            "0_good_#", "1_bad_#", "0_good_%", "1_bad_%",
            "WOE", "IV", "total", "total_pct", "bad_rate",
        ]+
        (["cum_1_bad_%", "cum_0_good_%", "cum_total_pct", "lift", "ks"] if with_lift_ks else [])+
        (["wls_adj_WOE", "_wls_adj_woe_details"] if with_wls_adj_woe else [])
    ]

    return crosstab

def func_woe_report_v2(
        in_df,
        var_name,
        target_label,
        with_total=True, good_label_val="0_good", bad_label_val="1_bad",
        floating_point=1e-4,
        with_lift_ks=False, lift_calc_ascending=True,
        with_wls_adj_woe=False,
    ):
    crosstab = func_woe_report_v1(
        in_var=in_df[var_name], in_target=in_df[target_label],
        with_total=with_total, good_label_val=good_label_val, bad_label_val=bad_label_val, floating_point=floating_point,
        with_lift_ks=with_lift_ks, lift_calc_ascending=lift_calc_ascending,
        with_wls_adj_woe=with_wls_adj_woe,
    )
    return crosstab


###########################################################################
# WOE组合图
def func_plot_woe(
        crosstab,
        plot_badrate=False,
        with_nan_info=True,
        with_wls_adj_woe=False,
    ):
    crosstab = crosstab[crosstab.index!="total"]
    if not with_nan_info:
        _cond = [
            not all(["NaN" in t for t in s0.split("_")[1].split(",")])
            for s0 in crosstab.index.tolist()
        ]
        crosstab = crosstab[_cond].copy(deep=True)
    
    fig = plt.figure(1, figsize=(12, 8))
    plt.xticks(rotation=90)

    ax = fig.add_subplot(111)
    ax.bar(crosstab.index, crosstab["total"], alpha=0.5, color="gray")
    ax2 = ax.twinx()
    ax2.grid(False)
    if plot_badrate==True:
        ax2.plot(
            crosstab.index,
            crosstab["bad_rate"],
            label="bad_rate",
            color="darkred",
            linewidth=2.5,
        )
    else:
        ax2.plot(
            crosstab.index,
            crosstab["WOE"],
            label="WOE",
            color="darkred",
            linewidth=2.5,
        )
        if with_wls_adj_woe==True:
            ax2.plot(
                crosstab.index,
                crosstab["wls_adj_WOE"],
                label="wls_adj_WOE",
                color="darkblue",
                linewidth=2.5,
            )
            ax2.plot(
                crosstab.index,
                crosstab["_wls_adj_woe_details"].apply(lambda s0: (np.NaN if pd.isna(s0) else s0.get("wls_p1_WOE"))),
                label="wls_adj_p1_woe",
                color="forestgreen",
                linestyle="-.",
                linewidth=1.5,
            )
            ax2.plot(
                crosstab.index,
                crosstab["_wls_adj_woe_details"].apply(lambda s0: (np.NaN if pd.isna(s0) else s0.get("wls_p2_WOE"))),
                label="wls_adj_p2_woe",
                color="darkgreen",
                linestyle="--",
                linewidth=1.5,
            )
    # plt.legend()
    plt.legend(loc=2)
    plt.show()


###########################################################################
# 分箱单调性统计
def func_calc_binning_woe_stats(in_crosstab, plot=False):
    crosstab = in_crosstab[in_crosstab.index!="total"]
    _cond = [
        not all(["NaN" in t for t in s0.split("_")[-1].split(",")])
        for s0 in crosstab.index.tolist()
    ]
    crosstab = crosstab[_cond].copy(deep=True)
    
    _data = crosstab["WOE"].tolist()
#     if len(_data)>=2:
#         _data = np.concatenate([[_data[_idx], np.mean([_data[_idx], _data[_idx+1]])] for _idx in range(len(_data)-1)]).tolist()+[_data[-1]]
    _data = pd.DataFrame(_data, columns=["WOE"])
    _data["WOE_lead"] = _data["WOE"].shift(periods=-1)
    _data["WOE_diff"] = _data["WOE"]-_data["WOE_lead"]
    _data["WOE_diff_lead"] = _data["WOE_diff"].shift(periods=-1)
    
    _data["const"] = 1
    _data["idx"] = [s0+1 for s0 in range(_data.shape[0])]
    _data["idx"] = _data["idx"]/max(_data["idx"])
#     _data["idx_2"] = _data["idx"]**2
#     _data["idx_3"] = _data["idx"]**3
    
    rt_totoal_pct_cv = crosstab["total_pct"].std()/crosstab["total_pct"].mean()
    rt_totoal_pct_max = crosstab["total_pct"].max()
    rt_corr_pearson = _data[["idx", "WOE"]].corr(method="pearson").values[0, 1]
    rt_corr_spearman = _data[["idx", "WOE"]].corr(method="spearman").values[0, 1]
    rt_corr_kendall = _data[["idx", "WOE"]].corr(method="kendall").values[0, 1]
#     rt_corr_pearson = [s0 for s0 in pearsonr(_data["idx"], _data["WOE"])]
#     rt_corr_spearman = [s0 for s0 in spearmanr(_data["idx"], _data["WOE"])]
#     rt_corr_kendall = [s0 for s0 in kendalltau(_data["idx"], _data["WOE"])]
    rt_zp_cnt = len([
        (s0, s1)
        for s0, s1 in _data[_data["WOE_diff_lead"].notna()][["WOE_diff", "WOE_diff_lead"]].values
        if s0*s1<0
    ])
    
#     _cols_ols = ["const", "idx_3", "idx_2", "idx"]
#     _exog = _data[_cols_ols]
#     _endog = _data["WOE"]
#     ols_model = sm.OLS(
#         endog=_endog,
#         exog=_exog,
#         hasconst=True,
#     )
#     ols_model_res = ols_model.fit(method="pinv")
#     _data["prediction"] = ols_model_res.predict(exog=_exog)
    
    rt = pd.Series(OrderedDict({
        "binning_cnt": int(in_crosstab[in_crosstab.index!="total"].shape[0]),
        "binning_cnt_notna": int(crosstab.shape[0]),
        "IV_notna": crosstab["IV"].sum(),
        "IV_pre_bin_notna": crosstab["IV"].sum()/int(crosstab.shape[0]),
        
        "totoal_pct_cv": rt_totoal_pct_cv,
        "total_pct_max": rt_totoal_pct_max,
        "corr_pearson": rt_corr_pearson,
        "corr_spearman": rt_corr_spearman,
        "corr_kendall": rt_corr_kendall,
        "zp_cnt": rt_zp_cnt,
        
#         "ols_resquared": ols_model_res.rsquared,
#         "ols_resquared_adj": ols_model_res.rsquared_adj,
#         "ols_param_const": ols_model_res.params["const"],
#         "ols_param_idx_3": ols_model_res.params["idx_3"],
#         "ols_param_idx_2": ols_model_res.params["idx_2"],
#         "ols_param_idx": ols_model_res.params["idx"],
#         "ols_pvalues_const": ols_model_res.pvalues["const"],
#         "ols_pvalues_idx_3": ols_model_res.pvalues["idx_3"],
#         "ols_pvalues_idx_2": ols_model_res.pvalues["idx_2"],
#         "ols_pvalues_idx": ols_model_res.pvalues["idx"],
    }))
    
    if plot:
        func_plot_woe(crosstab, plot_badrate=True, with_nan_info=True, with_wls_adj_woe=False)
#         print(ols_model_res.summary2())
    return rt


###########################################################################
# 计算KS
def func_calc_ks_cross(y_labels, y_pred, plot=False):
    y_labels = pd.Series(y_labels).values
    y_pred = pd.Series(y_pred).values
    
    crossfreq = pd.crosstab(index=y_pred, columns=y_labels)
    crossfreq.index.name = "predict_prob"
    crossfreq.columns.name = ""
    
    crossdens = crossfreq.cumsum(axis=0)/crossfreq.sum()
    crossdens['gap'] = abs(crossdens.iloc[:, 0]-crossdens.iloc[:, 1])
    ks = crossdens[crossdens['gap']==crossdens['gap'].max()]
    if plot:
        _data = pd.DataFrame(np.concatenate([
            y_labels.reshape([-1, 1]),
            y_pred.reshape([-1, 1]),
        ], axis=1), columns=["TARGET_label_bad", "PRED_bad_prob"]).astype(dtype={
            "TARGET_label_bad": np.str,
            "PRED_bad_prob": np.float,
        })
        sns.violinplot(data=_data, y="PRED_bad_prob", x="TARGET_label_bad", hue="TARGET_label_bad")
        
        crossdens.plot(kind="line")
        plt.show()
    return ks, crossdens


###########################################################################
# 计算AUC_ROC
def func_calc_auc_roc(y_labels, y_pred, plot=False):
    from sklearn import metrics
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_labels, y_pred)
    auc = metrics.auc(false_positive_rate, true_positive_rate)
    if plot:
        plt.style.use({"figure.figsize": [s0*3 for s0 in (2, 2)]})
        plt.title('ROC')
        plt.plot(false_positive_rate, true_positive_rate, color='b', label='AUC = {:0.4f}'.format(auc))
        plt.legend(loc='lower right')
        plt.plot([0,1], [0,1], 'r--')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.show()
        plt.style.use({"figure.figsize": [s0*3 for s0 in (4, 2)]})
        plt.show()
    return auc


###########################################################################
# 计算LIFT
def func_calc_lift(y_labels, y_pred, bucket_cnt=20, bad_label="bad", plot=False):
    y_pred.name = "predict"

    _t = y_pred.sort_values().reset_index()
    mapping_bucket_idx = dict([(_t.loc[idx, "index"], min(int(idx/(_t.shape[0]//bucket_cnt))+1, bucket_cnt)) for idx in _t.index.tolist()])
    y_gb = pd.Series(y_pred.index, name="gb").apply(mapping_bucket_idx.get)

    mapping_bucket_min_predict = dict(y_gb.reset_index().merge(right=y_pred.reset_index(), how="left", on=["index"]) \
        .groupby(by=["gb"]).apply(lambda s0: s0["predict"].min()))
    mapping_bucket_min_predict[1] = 0

    crosstab = pd.crosstab(
        index=y_gb,
        columns=y_labels,
    ).sort_index(ascending=False)
    crosstab.index.name = "gb"
    crosstab["obs_cnt"] = crosstab.sum(axis=1)
    crosstab = crosstab.reset_index()
    crosstab = crosstab[["gb", bad_label, "obs_cnt"]].rename(columns={bad_label: "bad_cnt"})

    crosstab["predict_prob_gte"] = crosstab["gb"].apply(mapping_bucket_min_predict.get)
    crosstab["bad_pct"] = crosstab["bad_cnt"]/crosstab["bad_cnt"].sum()
    crosstab["obs_pct"] = crosstab["obs_cnt"]/crosstab["obs_cnt"].sum()
    crosstab["bad_pct_cum"] = crosstab["bad_pct"].cumsum()
    crosstab["obs_pct_cum"] = crosstab["obs_pct"].cumsum()
    crosstab["lift"] = crosstab["bad_pct_cum"]/crosstab["obs_pct_cum"]
    crosstab["bad_rate"] = crosstab["bad_cnt"]/crosstab["obs_cnt"]

    if plot:
        crosstab.set_index(keys=["obs_pct_cum"])[["lift"]].plot(kind="line")
        crosstab.set_index(keys=["gb"])[["bad_pct", "obs_pct"]].plot(kind="bar")
        plt.show()
    return crosstab


###########################################################################
# 计算PSI

# 计算离散变量PSI
def func_calc_psi_discrete_v1(in_data_actual, in_data_expected, plot=False):
    in_data_actual = pd.Series(in_data_actual).fillna("NaN")
    in_data_expected = pd.Series(in_data_expected).fillna("NaN")
    
    psi_table = pd.merge(
        left=pd.DataFrame([{
            "data_label": _data_label,
            "actual_cnt": _cnt,
        } for _data_label, _cnt in in_data_actual.value_counts().sort_index().reset_index().values]),
        right=pd.DataFrame([{
            "data_label": _data_label,
            "expected_cnt": _cnt,
        } for _data_label, _cnt in in_data_expected.value_counts().sort_index().reset_index().values]),
        how="outer", on="data_label",
    )[["data_label", "actual_cnt", "expected_cnt"]]
    psi_table[["actual_pct", "expected_pct"]] = psi_table[["actual_cnt", "expected_cnt"]].apply(lambda s0: s0/s0.sum())
    psi_table["minus_act_exp"] = psi_table["actual_pct"] - psi_table["expected_pct"]
    psi_table["ln_act_exp"] = np.log((psi_table["actual_pct"]+1e-2)/(psi_table["expected_pct"]+1e-2))
    psi_table["Index"] = psi_table["minus_act_exp"]*psi_table["ln_act_exp"]

    psi = psi_table["Index"].sum()
    if plot:
        psi_table.set_index(keys=["data_label"])[["actual_pct", "expected_pct"]].plot(kind="bar")
        plt.show()
    return psi, psi_table

def func_calc_psi_discrete_v2(in_df, actual_label, expected_label, plot=False):
    psi, psi_table = func_calc_psi_discrete_v1(
        in_data_actual=in_df[actual_label],
        in_data_expected=in_df[expected_label],
        plot=plot)
    return psi, psi_table
    

# 计算连续变量PSI，等距分箱操作
def func_calc_psi_continuous_v1(in_data_actual, in_data_expected, bins_cnt=10, plot=False):
    in_data_actual = pd.Series(in_data_actual)
    in_data_expected = pd.Series(in_data_expected)
    
    _cut_left = in_data_expected.min()
    _cut_right = in_data_expected.max()
    cut_bins = [-np.inf]+[_cut_left+((_cut_right-_cut_left)/bins_cnt)*s0 for s0 in range(1, bins_cnt)]+[np.inf]
    in_data_actual_binning = func_binning_continuous_v1(
        in_data=in_data_actual,
        bins=cut_bins,
        out_type="01_info",
        right_border=True, include_lowest=True,
    )
    in_data_expected_binning = func_binning_continuous_v1(
        in_data=in_data_expected,
        bins=cut_bins,
        out_type="01_info",
        right_border=True, include_lowest=True,
    )
    
    psi, psi_table = func_calc_psi_discrete_v1(
        in_data_actual=in_data_actual_binning, in_data_expected=in_data_expected_binning, plot=plot)
    return psi, psi_table

def func_calc_psi_continuous_v2(in_df, actual_label, expected_label, bins_cnt=10, plot=False):
    psi, psi_table = func_calc_psi_continuous_v1(
        in_data_actual=in_df[actual_label],
        in_data_expected=in_df[expected_label],
        bins_cnt=bins_cnt,
        plot=plot)
    return psi, psi_table


# 按月度频率计算离散变量PSI
def func_calc_psi_discrete_features_monthly(in_data_features, in_data_YM, base_last_months=4, verbose=0):
    in_data_features = in_data_features.reset_index(drop=True)
    in_data_YM = pd.DataFrame(pd.Series(in_data_YM).values, columns=["data_dt"])
    _data = pd.merge(left=in_data_YM, right=in_data_features,
                     how="left", left_index=True, right_index=True)
    feature_names = in_data_features.columns.tolist()
    data_dt = in_data_YM["data_dt"].drop_duplicates().sort_values().tolist()
    mapping_data_dt_for_calc = dict(
        [(s0, [t for t in data_dt if t<s0][-base_last_months:]) for s0 in data_dt[1:]]
    )
    mapping_gb_data_dt = dict([(_dt, _df[feature_names]) for _dt, _df in _data.groupby(by=["data_dt"])])
    
    _features_psi = []
    if verbose:
        print("========================================")
        print("{} features pending...".format(len(feature_names)))
    for _idx, _feature_name in enumerate(feature_names):
        _features_psi.append(OrderedDict(
            [("feature_name", _feature_name)]+
            [(_dt, OrderedDict([(s0,
                     func_calc_psi_discrete_v1(
                         in_data_actual=mapping_gb_data_dt.get(_dt)[_feature_name],
                         in_data_expected=mapping_gb_data_dt.get(s0)[_feature_name],
                         plot=False)[0]
                     ) for s0 in _base_dts]))
             for _dt, _base_dts in mapping_data_dt_for_calc.items()][:]
        ))
        if verbose:
            print("[done {}] {}".format(_idx+1, _feature_name))
    df_features_psi = pd.DataFrame(_features_psi).set_index(keys=["feature_name"])
    df_features_psi_summary = df_features_psi.applymap(lambda s0: pd.Series(s0).mean())
    return df_features_psi_summary


# 相关系数矩阵热力图
def func_correlation_matrix_hotmap(df, figsize=12):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure(figsize=(figsize, figsize))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Abalone Feature Correlation')
    labels = df.columns
    ax1.set_xticklabels([], fontsize=10)
    ax1.set_yticklabels(labels, fontsize=10)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    plt.show()



if __name__=="__main__":
    pass

