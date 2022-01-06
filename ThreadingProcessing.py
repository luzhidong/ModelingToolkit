import os
import re
import sys
import csv
import json
import time
import pytz
import datetime
strptime = datetime.datetime.strptime
strftime = datetime.datetime.strftime
from dateutil.relativedelta import relativedelta
from dateutil import rrule

from collections import OrderedDict
from itertools import product
import pickle

import gc
import multiprocessing

import threading
import queue

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import random


class Threading_Proc(threading.Thread):
    
    def __init__(
        self,
        thread_name,
        pending_queue,
        lock,
        func_processing,
        daemon=True,
        ):
        
        threading.Thread.__init__(self, daemon=daemon)
        self.thread_name = thread_name
        self.pending_queue = pending_queue
        self.lock = lock
        self.func_processing = func_processing
        
        self.out = []
        
    def func_Proc(self):
        
        while self.pending_queue.qsize()>0:
            with self.lock:
                args_data = self.pending_queue.get()
            
            #########################################
            print(
                "[{}] thread_name: {}, pending_queue_size: {}, args_data: {}".format(
                    strftime(datetime.datetime.now(), "%Y-%d-%m %H:%M:%S"),
                    self.thread_name,
                    self.pending_queue.qsize(),
                    args_data,
                )
            )
            
            #########################################
            if self.func_processing is not None:
                self.out.append(self.func_processing(args_data))
                
            else:
                pass
            
            #########################################
            time.sleep(np.random.uniform()+1)
        
    def run(self):
        
        print("--------------------------------------")
        print(">>>START<<<  {}.".format(self.thread_name))
        print("--------------------------------------")
        print()
        self.func_Proc()
        print()
        print("--------------------------------------")
        print(">>>EXIT<<<  {}.".format(self.thread_name))
        print("--------------------------------------")

def func_threading_processing(
    threading_n,
    pending_queue,
    func_processing=None,
    ):
    lock = threading.RLock()
    # lock = threading.Lock()
    
    threading_list = [
        Threading_Proc(
            thread_name="Thread-{}".format(s0),
            pending_queue=pending_queue,
            lock=lock,
            func_processing=func_processing,
        )
        for s0 in range(threading_n)
    ]
    
    for t in threading_list:
        t.start()
    for t in threading_list:
        t.join()
    
    rt = [{s0.thread_name: s0.out} for s0 in threading_list]
    return rt, threading_list






