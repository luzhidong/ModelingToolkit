import os
import sys
import time
import datetime
import pickle
import re
import json
import urllib
import hashlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('max_colwidth', 200)
pd.set_option('max_columns', None)
pd.set_option('display.width', 1000)


# os.chdir(r'C:\\Users\\luzhidong-alienware\\PycharmProjects\\MyProj\\Prj_801_quant')
# os.chdir(r'/root/MyProj/quant/')

# import warnings
# warnings.filterwarnings('ignore')


################################################################################
def func_df2list(input_df, bulk_size=None, func_json_trans_processing=None):
    input_df_size = input_df.shape[0]
    input_df = input_df.reset_index(drop=True)
    
    if bulk_size is None:
        bulk_size = 1
    for _idx_bulk in range(0, input_df_size//bulk_size+1):
        _s_bulk = bulk_size*_idx_bulk
        _e_bulk = bulk_size*(_idx_bulk+1)
        _bulk_df = input_df.iloc[_s_bulk:_e_bulk, :]
        
        rt = [dict(zip(input_df.columns, s0)) for s0 in _bulk_df.values]
        rt = [dict([(k, v) for k, v in s0.items() if not pd.isnull(v)]) for s0 in rt]
        
        # func_json_trans_processing
        if func_json_trans_processing:
            rt = list(map(func_json_trans_processing, rt))
        yield rt


def func_curl(url, method='GET', headers=None, json_data=None, warnning_print=False):
    if headers is None:
        headers = {
            "Content-Type": "application/json",
        }
    if isinstance(json_data, dict):
        encoded_data = json.dumps(obj=json_data).encode()
    elif isinstance(json_data, list):
        encoded_data = '\n'.join([json.dumps(s0) for s0 in json_data]) + '\n'
        encoded_data = encoded_data.encode()
    else:
        encoded_data = None
    
    req = urllib.request.Request(url=url,
                                 data=encoded_data,
                                 headers=headers,
                                 method=method)
    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        rt = json.loads(s=result.decode())
    except Exception as e:
        result = b''
        if warnning_print:
            print(e)
        rt = None
    return rt


################################################################################
def es_import_data_bulk(input_df, es_url_base, index,
                        func_json_trans_processing, set_id_field,
                        bulk_size=10000, op_type='create', log_print=True):
    url = '{}/{}/_bulk'.format(es_url_base, index)
    input_data_size = input_df.shape[0]
    bulk_size = min(bulk_size, input_data_size)
    
    _idx_bulk = 0
    _time = time.time()
    data_generator = func_df2list(input_df=input_df, bulk_size=bulk_size, func_json_trans_processing=func_json_trans_processing)
    while True:
        try:
            _bulk_json_data = data_generator.__next__()
            if len(_bulk_json_data)==0:
                break
        except:
            print('done')
            break
        
        bulk_json_data = []
        for _d in _bulk_json_data:
            _metadata = {op_type: {"_id": hashlib.md5((''.join([str(_d[k]) for k in set_id_field])).encode()).hexdigest()}}
            bulk_json_data.append(_metadata)
            bulk_json_data.append(_d)
        res = func_curl(url=url,
                        method='POST',
                        json_data=bulk_json_data)
        if res is None:
            print("[Import Faild]")
#         else:
#             print("[Import Successed] reccord size {}".format(len(_bulk_json_data)))
            
        if log_print:
            print("[{}] [{}_{}_{}%]".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                            input_data_size//bulk_size, _idx_bulk, '{:.2f}'.format(_idx_bulk/(input_data_size//bulk_size)*100)))
        _idx_bulk = _idx_bulk+1



def es_get_data(es_url_base, index,
                query_dict=None, bulk_size=None, scroll_keep_time='1m',
                warnning_print=False):
    url = '{}/{}/_search?scroll={}'.format(es_url_base, index, scroll_keep_time)
    url_scroll = '{}/_search/scroll'.format(es_url_base)
    if query_dict is None and bulk_size is not None:
        query_dict = {
            "size": bulk_size,
        }
    elif query_dict.get("size") is not None and bulk_size is not None:
        query_dict['size'] = bulk_size
    
    i = 0
    while True:
        i = i+1
        ######################################################
        if i==1:
            _d = func_curl(url=url, method='POST', json_data=query_dict,
                           warnning_print=warnning_print)
            _scroll_id = _d['_scroll_id']
            scroll_dict = {
                "scroll" : scroll_keep_time,
                "scroll_id": _scroll_id,
            }
        else:
            _d = func_curl(url=url_scroll, method='POST', json_data=scroll_dict,
                           warnning_print=warnning_print)
        ######################################################
        _data = _d['hits']['hits']
        yield _data
        
        
        
def es_get_data_sql(es_url_base, sql_string, fetch_size=None,
                    datatype_es2df=None,
                    timeout=120, warnning_print=False):
    url = "{}/_sql?format=json".format(es_url_base)
    query_dict = {
        "query": sql_string,
        "fetch_size": (fetch_size if fetch_size is not None else 1000),
        "request_timeout": "{}s".format(timeout),
        "page_timeout": "{}s".format(timeout),
    }
    datatype_es2df_0 = {
        "text": np.str,
        "keyword": np.str,
        "long": np.int32,
        "integer": np.int32,
        "short": np.int32,
        "byte": np.int32,
        "double": np.float32,
        "float": np.float32,
        "half_float": np.float32,
        "scaled_float": np.float32,
        "datetime": np.datetime64,
        "date": np.datetime64,
        "date_nanos": np.datetime64,
        "boolean": np.str,
    }
    if datatype_es2df is not None and isinstance(datatype_es2df, dict):
        datatype_es2df_0.update(datatype_es2df.items())
    datatype_es2df = datatype_es2df_0
    
    i = 0
    while True:
        i = i+1
        ######################################################
        if i==1:
            _d = func_curl(url=url,
                           method="POST", json_data=query_dict,
                           warnning_print=warnning_print)
            _columns = _d.get('columns')
            _columns = [dict(list(s0.items())+[("dtype", datatype_es2df.get(s0['type'], np.str))]) for s0 in _columns]
            _cursor = _d.get('cursor')
        else:
            query_dict = {
                "cursor": _cursor,
                "request_timeout": "{}s".format(timeout),
                "page_timeout": "{}s".format(timeout),
            }
            _d = func_curl(url=url,
                           method="POST", json_data=query_dict,
                           warnning_print=warnning_print)
        ######################################################
        
        # _df = pd.DataFrame(data=[], columns=[s0['name'] for s0 in _columns], dtype=np.str)
        if _d is not None and len(_d.get('rows'))!=0:
            _df = pd.DataFrame(data=_d.get('rows'), columns=[s0['name'] for s0 in _columns])
        else:
            _df = pd.DataFrame(data=[], columns=[s0['name'] for s0 in _columns])
            
        _df = _df.astype(dict([(s0['name'], s0['dtype']) for s0 in _columns]))
        yield _df
        
        
################################################################################
def func_json_trans_processing_price(input_json):
    rt = {
        'ts_code': input_json.get('ts_code'),
        'ts_code_type': input_json.get('ts_code_type'),
        'trade_time': input_json.get('trade_time'),
        'price': {
            'open': input_json.get('open'),
            'high': input_json.get('high'),
            'low': input_json.get('low'),
            'close': input_json.get('close'),
            'vol': input_json.get('vol'),
            'amount': input_json.get('amount'),
            'adj_factor': input_json.get('adj_factor'),
        }
    }
    return rt

def func_json_trans_processing_f_report(input_json):
    rt = {
        'code': input_json.get('code'),
        'name': input_json.get('name'),
        'report_date': input_json.get('report_date'),
        'year': input_json.get('year'),
        'quarter': input_json.get('quarter'),
        'report': {
            'eps': input_json.get('eps'),
            'eps_yoy': input_json.get('eps_yoy'),
            'bvps': input_json.get('bvps'),
            'roe': input_json.get('roe'),
            'epcf': input_json.get('epcf'),
            'net_profits': input_json.get('net_profits'),
            'profits_yoy': input_json.get('profits_yoy'),
            'distrib': input_json.get('distrib'),
        }
    }
    return rt

def func_json_trans_processing_f_profit(input_json):
    rt = {
        'code': input_json.get('code'),
        'name': input_json.get('name'),
        'year': input_json.get('year'),
        'quarter': input_json.get('quarter'),
        'profit': {
            'roe': input_json.get('roe'),
            'net_profit_ratio': input_json.get('net_profit_ratio'),
            'gross_profit_rate': input_json.get('gross_profit_rate'),
            'net_profits': input_json.get('net_profits'),
            'eps': input_json.get('eps'),
            'business_income': input_json.get('business_income'),
            'bips': input_json.get('bips'),
        }
    }
    return rt

def func_json_trans_processing_f_operation(input_json):
    rt = {
        'code': input_json.get('code'),
        'name': input_json.get('name'),
        'year': input_json.get('year'),
        'quarter': input_json.get('quarter'),
        'operation': {
            'arturnover': input_json.get('arturnover'),
            'arturndays': input_json.get('arturndays'),
            'inventory_turnover': input_json.get('inventory_turnover'),
            'inventory_days': input_json.get('inventory_days'),
            'currentasset_turnover': input_json.get('currentasset_turnover'),
            'currentasset_days': input_json.get('currentasset_days'),
        }
    }
    return rt

def func_json_trans_processing_f_growth(input_json):
    rt = {
        'code': input_json.get('code'),
        'name': input_json.get('name'),
        'year': input_json.get('year'),
        'quarter': input_json.get('quarter'),
        'growth': {
            'mbrg': input_json.get('mbrg'),
            'nprg': input_json.get('nprg'),
            'nav': input_json.get('nav'),
            'targ': input_json.get('targ'),
            'epsg': input_json.get('epsg'),
            'seg': input_json.get('seg'),
        }
    }
    return rt

def func_json_trans_processing_f_cashflow(input_json):
    rt = {
        'code': input_json.get('code'),
        'name': input_json.get('name'),
        'year': input_json.get('year'),
        'quarter': input_json.get('quarter'),
        'cashflow': {
            'cf_sales': input_json.get('cf_sales'),
            'rateofreturn': input_json.get('rateofreturn'),
            'cf_nm': input_json.get('cf_nm'),
            'cf_liabilities': input_json.get('cf_liabilities'),
            'cashflowratio': input_json.get('cashflowratio'),
        }
    }
    return rt

def func_json_trans_processing_f_debtpaying(input_json):
    def cast_float(in_val):
        try:
            rt = float(in_val)
        except:
            rt = None
        return rt
    rt = {
        'code': input_json.get('code'),
        'name': input_json.get('name'),
        'year': input_json.get('year'),
        'quarter': input_json.get('quarter'),
        'debtpaying': {
            'currentratio': cast_float(input_json.get('currentratio')),
            'quickratio': cast_float(input_json.get('quickratio')),
            'cashratio': cast_float(input_json.get('cashratio')),
            'icratio': cast_float(input_json.get('icratio')),
            'sheqratio': cast_float(input_json.get('sheqratio')),
            'adratio': cast_float(input_json.get('adratio')),
        }
    }
    return rt

def func_json_trans_processing_tick(input_json):
    rt = {
        'ts_code': input_json.get('ts_code'),
        'datetime': input_json.get('datetime'),
        'price': input_json.get('price'),
        'change': input_json.get('change'),
        'volume': input_json.get('volume'),
        'amount': input_json.get('amount'),
        'type': input_json.get('type'),        
    }
    return rt









