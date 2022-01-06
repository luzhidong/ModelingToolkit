# -*- coding: utf-8 -*-
# 
#----------------------------------------------------------
# script name: 
#----------------------------------------------------------
# creator: zhidong.lu
# create date: 2019-05-30
# update date: 2019-05-30
# version: 1.0
#----------------------------------------------------------
# 
# 
#----------------------------------------------------------


# from __future__ import print_function
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import \
    interact, interactive, fixed, interact_manual, \
    Layout

import numpy as np
import pandas as pd

# pd.set_option('display.width', 500)
# pd.set_option('max_colwidth', 200)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 100)
# pd.set_option('display.min_rows', 20)

import warnings
warnings.filterwarnings('ignore')


#############################################################################################
# widget_Selection
# 清单选择器
class widget_Selection(object):
    def __init__(
            self,
            items_list,
            items_list_selected=[],
        ):
        
        ####################################################
        # 原始变量赋值保存
        self._items_list_ori = items_list
        self._items_list_selected_ori = items_list_selected
        
        ####################################################
        # 变量初始化
        self.initial_VARIABLE()
        
        ####################################################
        # widget初始化
        self.initial_WIDGET()
        
        ####################################################
        # event初始化
        self.initial_EVENT()
        
        ####################################################
        # main widget
        self.contruct_main_widget()
        
    
    ####################################################
    # 变量初始化
    def initial_VARIABLE(self):
        
        self.V_ITEMS_LIST = [s0 for s0 in self._items_list_ori if s0 not in self._items_list_selected_ori]
        self.V_ITEMS_LIST_SELECTED = self._items_list_selected_ori
    
    
    ####################################################
    # widget初始化
    def initial_WIDGET(self):
        
        ##########################
        self.WIDGET_SELECTMULTIPLE_LEFT = widgets.SelectMultiple(
            options=self.V_ITEMS_LIST,
            value=[],
            rows=10,
            description="",
            # style={"description_width": "initial"},
            disabled=False,
        )
        
        ##########################
        self.WIDGET_SELECTMULTIPLE_RIGHT= widgets.SelectMultiple(
            options=self.V_ITEMS_LIST_SELECTED,
            value=[],
            rows=10,
            description="",
            # style={"description_width": "initial"},
            disabled=False,
        )
        
        ##########################
        self.WIDGET_BUTTON_ADD = widgets.Button(description="Add")
        self.WIDGET_BUTTON_REMOVE = widgets.Button(description="Remove")
        self.WIDGET_BUTTON_ADD_ALL = widgets.Button(description="Add ALL")
        self.WIDGET_BUTTON_RESET = widgets.Button(description="Reset")
        
        ##########################
        self.WIDGET_BUTTON_UP = widgets.Button(description="Up")
        self.WIDGET_BUTTON_DOWN = widgets.Button(description="Down")
        
        ##########################
        self.WIDGET_BUTTON_SAVE = widgets.Button(description="Save")
        
        ##########################
        self.WIDGET_ALTER_STATUS = widgets.Label("{}".format("no changed"))
        
    
    ####################################################
    def contruct_main_widget(self):
        
        self.WIDGET_DASHBOARD = widgets.Box(
            children=[
                ##########################
                widgets.Box(
                    children=[
                        self.WIDGET_BUTTON_ADD,
                        self.WIDGET_BUTTON_REMOVE,
                        self.WIDGET_BUTTON_ADD_ALL,
                        self.WIDGET_BUTTON_RESET,
                    ],
                    layout=widgets.Layout(
                        display="flex",
                        flex_flow="row",
                        width="750px",
                        height="50px",
                        justify_content="center",
                        align_items="center",
                    ),
                ),
                ##########################
                widgets.Box(
                    children=[
                        self.WIDGET_BUTTON_UP,
                        self.WIDGET_BUTTON_DOWN,
                    ],
                    layout=widgets.Layout(
                        display="flex",
                        flex_flow="row",
                        width="750px",
                        height="50px",
                        justify_content="center",
                        align_items="center",
                    ),
                ),
                ##########################
                widgets.Box(
                    children=[
                        self.WIDGET_BUTTON_SAVE,
                        self.WIDGET_ALTER_STATUS,
                    ],
                    layout=widgets.Layout(
                        display="flex",
                        flex_flow="row",
                        width="750px",
                        height="50px",
                        justify_content="center",
                        align_items="center",
                    ),
                ),
                ##########################
                widgets.Box(
                    children=[
                        self.WIDGET_SELECTMULTIPLE_LEFT,
                        self.WIDGET_SELECTMULTIPLE_RIGHT,
                    ],
                    layout=widgets.Layout(
                        display="flex",
                        flex_flow="row",
                        width="750px",
                        height="250px",
                        justify_content="center",
                        # align_items="center",
                    ),
                ),
            ],
            layout=widgets.Layout(
                display="flex",
                flex_flow="column",
                border="grey 2px solid",
                width="800px",
                # height_max="400px",
                justify_content="center",
                align_items="center",
            ),
        )
        
    
    ####################################################
    # event初始化
    def initial_EVENT(self):
        
        ##########################
        # WIDGET_SELECTMULTIPLE_RIGHT
        self.WIDGET_SELECTMULTIPLE_RIGHT.observe(
            handler=self.EVENT_ON_VALUE_CHANGE_OF_SELECTMULTIPLE_RIGHT,
            names="options",
        )
        
        ##########################
        # WIDGET_BUTTON
        self.WIDGET_BUTTON_ADD.on_click(
            callback=lambda s0: self.EVENT_ON_CLICK_OF_BUTTON(action="Add"),
        )
        self.WIDGET_BUTTON_REMOVE.on_click(
            callback=lambda s0: self.EVENT_ON_CLICK_OF_BUTTON(action="Remove"),
        )
        self.WIDGET_BUTTON_ADD_ALL.on_click(
            callback=lambda s0: self.EVENT_ON_CLICK_OF_BUTTON(action="Add_All"),
        )
        self.WIDGET_BUTTON_RESET.on_click(
            callback=lambda s0: self.EVENT_ON_CLICK_OF_BUTTON(action="Reset"),
        )
        self.WIDGET_BUTTON_UP.on_click(
            callback=lambda s0: self.EVENT_ON_CLICK_OF_BUTTON(action="Up"),
        )
        self.WIDGET_BUTTON_DOWN.on_click(
            callback=lambda s0: self.EVENT_ON_CLICK_OF_BUTTON(action="Down"),
        )
        self.WIDGET_BUTTON_SAVE.on_click(
            callback=lambda s0: self.EVENT_ON_CLICK_OF_BUTTON(action="Save"),
        )
        
    
    ####################################################
    def EVENT_ON_VALUE_CHANGE_OF_SELECTMULTIPLE_RIGHT(self, change):
        
        self.WIDGET_ALTER_STATUS.value = "changed, no saved"
        
    
    ####################################################
    def EVENT_ON_CLICK_OF_BUTTON(self, action, d=None):
        
        if action=="Add":
            for _item in self.WIDGET_SELECTMULTIPLE_LEFT.value:
                if _item in self.WIDGET_SELECTMULTIPLE_LEFT.options:
                    self.WIDGET_SELECTMULTIPLE_LEFT.options = tuple([s0 for s0 in self.WIDGET_SELECTMULTIPLE_LEFT.options if s0!=_item])
                    self.WIDGET_SELECTMULTIPLE_RIGHT.options = tuple([s0 for s0 in self.WIDGET_SELECTMULTIPLE_RIGHT.options]+[_item])
                    self.WIDGET_ALTER_STATUS.value = "changed, no saved"
            
        elif action=="Remove":
            for _item in self.WIDGET_SELECTMULTIPLE_RIGHT.value:
                if _item in self.WIDGET_SELECTMULTIPLE_RIGHT.options:
                    self.WIDGET_SELECTMULTIPLE_RIGHT.options = tuple([s0 for s0 in self.WIDGET_SELECTMULTIPLE_RIGHT.options if s0!=_item])
                    self.WIDGET_SELECTMULTIPLE_LEFT.options = tuple([s0 for s0 in self.WIDGET_SELECTMULTIPLE_LEFT.options]+[_item])
                    self.WIDGET_ALTER_STATUS.value = "changed, no saved"
            
        elif action=="Add_All":
            for _item in self.WIDGET_SELECTMULTIPLE_LEFT.options:
                self.WIDGET_SELECTMULTIPLE_LEFT.options = tuple([s0 for s0 in self.WIDGET_SELECTMULTIPLE_LEFT.options if s0!=_item])
                self.WIDGET_SELECTMULTIPLE_RIGHT.options = tuple([s0 for s0 in self.WIDGET_SELECTMULTIPLE_RIGHT.options]+[_item])
                self.WIDGET_ALTER_STATUS.value = "changed, no saved"
            
        elif action=="Reset":
            self.initial_VARIABLE()
            self.WIDGET_SELECTMULTIPLE_LEFT.options = self.V_ITEMS_LIST
            self.WIDGET_SELECTMULTIPLE_RIGHT.options = self.V_ITEMS_LIST_SELECTED
            
            self.WIDGET_ALTER_STATUS.value = "reseted"
            self.WIDGET_SELECTMULTIPLE_LEFT.value = []
            self.WIDGET_SELECTMULTIPLE_RIGHT.value = []
            
        elif action=="Up":
            _options = self.WIDGET_SELECTMULTIPLE_RIGHT.options
            _value = self.WIDGET_SELECTMULTIPLE_RIGHT.value
            if len(_value)!=0:
                _idx = max(_options.index(_value[0]), 1)-1
                _options_moved = \
                    list(_options[0:_idx])+ \
                    list(_value)+ \
                    [s0 for s0 in _options[_idx:] if s0 not in _value]
                self.WIDGET_SELECTMULTIPLE_RIGHT.options = tuple(_options_moved)
            
            self.WIDGET_SELECTMULTIPLE_RIGHT.value = _value
            
        elif action=="Down":
            _options = self.WIDGET_SELECTMULTIPLE_RIGHT.options
            _value = self.WIDGET_SELECTMULTIPLE_RIGHT.value
            if len(_value)!=0:
                _idx = _options.index(_value[-1])+1
                _options_moved = \
                    [s0 for s0 in _options[0:_idx+1] if s0 not in _value]+ \
                    list(_value)+ \
                    list(_options[_idx+1:])
                self.WIDGET_SELECTMULTIPLE_RIGHT.options = tuple(_options_moved)
            
            self.WIDGET_SELECTMULTIPLE_RIGHT.value = _value
            
        elif action=="Save":
            self.V_ITEMS_LIST_SELECTED = list(self.WIDGET_SELECTMULTIPLE_RIGHT.options)
            
            self.WIDGET_ALTER_STATUS.value = "saved"
            self.WIDGET_SELECTMULTIPLE_LEFT.value = []
            self.WIDGET_SELECTMULTIPLE_RIGHT.value = []
    

#############################################################################################
# widget_PageController
# 页面控制器
class widget_PageController(object):
    def __init__(
            self,
            df,
            page_no_current=0,
            page_size=20,
            display=True,
        ):
        
        ####################################################
        # 原始变量赋值保存
        self._df_ori = df
        self._page_no_current_ori = page_no_current
        self._page_size_ori = page_size
        self._display_ori = display
        
        ####################################################
        # 变量初始化
        self.initial_VARIABLE()
        
        ####################################################
        # widget初始化
        self.initial_WIDGET()
        
        ####################################################
        # event初始化
        self.initial_EVENT()
        
        ####################################################
        # main widget
        self.contruct_main_widget()
        
    
    ####################################################
    # 变量初始化
    def initial_VARIABLE(self):
        
        self.df = self._df_ori
        self.display = self._display_ori
        
        self.V_PAGE_NO_CURRENT = self._page_no_current_ori
        self.V_PAGE_SIZE = self._page_size_ori
        self.V_TOTAL_SIZE = self._df_ori.shape[0]
        self.update_VARIABLE()
        
        self.df_out = pd.DataFrame([], columns=self._df_ori.columns)
        
    
    ####################################################
    # 变量更新
    def update_VARIABLE(self):
        
        self.V_PAGE_CNT = self.V_TOTAL_SIZE//self.V_PAGE_SIZE+1
        
        
    ####################################################
    # widget初始化
    def initial_WIDGET(self):
        
        ##########################
        self.WIDGET_SELECTION_PAGE_SIZE = widgets.Dropdown(
            options=[5, 10, 20, 30, 40 ,50],
            value=self.V_PAGE_SIZE,
            # rows=10,
            description="Page Size: ",
            disabled=False,
        )
        
        ##########################
        self.WIDGET_PAGE_NO_BAR = widgets.IntSlider(
            value=0,
            min=0,
            max=self.V_PAGE_CNT,
            step=1,
            description="Page No: ",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
        )
        
        ##########################
        self.WIDGET_BUTTON_PREVIOUS = widgets.Button(
            description="{}".format("previous"),
            layout={"width": "100px", "height": "25x"},
            disabled=False,
        )
        
        ##########################
        self.WIDGET_BUTTON_NEXT = widgets.Button(
            description="{}".format("next"),
            layout={"width": "70px", "height": "25x"},
            disabled=False,
        )
        
        ##########################
        self.WIDGET_PAGE_NO_BAR_STATUS = widgets.Label(
            "Page Size: {}, Current Page No: {}, Total Page: {}.".format(
                self.V_PAGE_SIZE,
                self.V_PAGE_NO_CURRENT,
                self.V_PAGE_CNT,
            )
        )
        
        ##########################
        self.WIDGET_BUTTON_APPLY = widgets.Button(description="Apply")
        
        ##########################
        self.OUTPUT_DF = widgets.Output()
        
    
    ####################################################
    def contruct_main_widget(self):
        
        self.WIDGET_DASHBOARD = widgets.Box(
            children=[
                ##########################
                widgets.Box(
                    children=[
                        self.WIDGET_SELECTION_PAGE_SIZE,
                        self.WIDGET_PAGE_NO_BAR,
                        self.WIDGET_BUTTON_PREVIOUS,
                        self.WIDGET_BUTTON_NEXT,
                    ],
                    layout=widgets.Layout(
                        display="flex",
                        flex_flow="row",
                        # border="grey 2px solid",
                        width="800px",
                        height="50px",
                        justify_content="flex-end",
                        align_items="center",
                    ),
                ),
                ##########################
                widgets.Box(
                    children=[
                        self.WIDGET_PAGE_NO_BAR_STATUS,
                        self.WIDGET_BUTTON_APPLY,
                    ],
                    layout=widgets.Layout(
                        display="flex",
                        flex_flow="row",
                        # border="grey 2px solid",
                        width="800px",
                        height="50px",
                        justify_content="flex-end",
                        align_items="center",
                    ),
                ),
                ##########################
                widgets.Box(
                    children=[
                        self.OUTPUT_DF,
                    ],
                    layout=widgets.Layout(
                        display="flex",
                        flex_flow="column",
                        border="grey 2px solid",
                        width="800px",
                        height_max="60%",
                        justify_content="flex-start",
                    ),
                ),
            ],
            layout=widgets.Layout(
                display="flex",
                flex_flow="column",
                border="grey 2px solid",
                width="1000px",
                height_max="1200px",
                # justify_content="flex-start",
                # align_items="center",
            ),
        )
        
    
    ####################################################
    # event初始化
    def initial_EVENT(self):
        
        ##########################
        # WIDGET_SELECTION_PAGE_SIZE
        self.WIDGET_SELECTION_PAGE_SIZE.observe(
            handler=self.EVENT_ON_VALUE_CHANGE_OF_SELECTION_PAGE_SIZE,
            names="value",
        )
        
        ##########################
        # WIDGET_PAGE_NO_BAR
        self.WIDGET_PAGE_NO_BAR.observe(
            handler=self.EVENT_ON_VALUE_CHANGE_OF_PAGE_NO_BAR,
            names="value",
        )
        
        ##########################
        # WIDGET_BUTTON_PREVIOUS
        self.WIDGET_BUTTON_PREVIOUS.on_click(
            callback=lambda s0: self.EVENT_ON_CLICK_OF_BUTTON_PAGE_NO(action="previous"),
        )
        
        ##########################
        # WIDGET_BUTTON_NEXT
        self.WIDGET_BUTTON_NEXT.on_click(
            callback=lambda s0: self.EVENT_ON_CLICK_OF_BUTTON_PAGE_NO(action="next"),
        )
        
        ##########################
        # WIDGET_BUTTON_APPLY
        self.WIDGET_BUTTON_APPLY.on_click(
            callback=self.EVENT_ON_CLICK_OF_BUTTON_APPLY,
        )
        
    
    ####################################################
    def EVENT_ON_VALUE_CHANGE_OF_SELECTION_PAGE_SIZE(self, change):
        
        self.V_PAGE_NO_CURRENT = self.WIDGET_PAGE_NO_BAR.value
        self.V_PAGE_SIZE = self.WIDGET_SELECTION_PAGE_SIZE.value
        
        self.update_VARIABLE()
        self.WIDGET_PAGE_NO_BAR.max = self.V_PAGE_CNT
        self.WIDGET_PAGE_NO_BAR_STATUS.value = "Page Size: {}, Current Page No: {}, Total Page: {}.".format(
            self.V_PAGE_SIZE,
            self.V_PAGE_NO_CURRENT,
            self.V_PAGE_CNT,
        )
        
    
    ####################################################
    def EVENT_ON_VALUE_CHANGE_OF_PAGE_NO_BAR(self, change):
        
        self.V_PAGE_NO_CURRENT = self.WIDGET_PAGE_NO_BAR.value
        self.WIDGET_PAGE_NO_BAR_STATUS.value = "Page Size: {}, Current Page No: {}, Total Page: {}.".format(
            self.V_PAGE_SIZE,
            self.V_PAGE_NO_CURRENT,
            self.V_PAGE_CNT,
        )
        
    
    ####################################################
    def EVENT_ON_CLICK_OF_BUTTON_PAGE_NO(self, action):
        
        if action=="previous":
            self.WIDGET_PAGE_NO_BAR.value = max(self.WIDGET_PAGE_NO_BAR.value-1, 0)
            self.V_PAGE_NO_CURRENT = self.WIDGET_PAGE_NO_BAR.value
            self.WIDGET_BUTTON_APPLY.click()
            
        elif action=="next":
            self.WIDGET_PAGE_NO_BAR.value = min(self.WIDGET_PAGE_NO_BAR.value+1, self.V_PAGE_CNT)
            self.V_PAGE_NO_CURRENT = self.WIDGET_PAGE_NO_BAR.value
            self.WIDGET_BUTTON_APPLY.click()
        
    
    ####################################################
    def EVENT_ON_CLICK_OF_BUTTON_APPLY(self, action):
        
        #############################################################################
        self.df = self._df_ori \
            .reset_index(drop=False).rename(columns={"index": "_page_no"}) \
            .reset_index(drop=False).rename(columns={"index": "_page_idx"}) \
            
        self.df["_page_no"] = self.df["_page_no"].apply(lambda s0: s0//self.V_PAGE_SIZE)
        self.df["_page_idx"] = self.df["_page_idx"].apply(lambda s0: np.mod(s0, self.V_PAGE_SIZE))
        self.df_out = self.df \
            .query("_page_no=='{}'".format(self.WIDGET_PAGE_NO_BAR.value)) \
            .drop(labels=["_page_idx", "_page_no"], axis=1)
        
        #############################################################################
        if self.display:
            self.OUTPUT_DF.clear_output()
            with self.OUTPUT_DF:
                display(widgets.Label("-----------------------------query result-----------------------------"))
                display(widgets.Label("{} rows, {} cols.".format(
                    self.df_out.shape[0],
                    self.df_out.shape[1],
                )))
                display(self.df_out)
    
    
#############################################################################################
# widget_OrderSetting
# 字段排序设置
class widget_OrderSetting(object):
    def __init__(
            self,
            items_list,
            items_list_selected=[],
            items_list_selected_ascending=[],
        ):
        
        ####################################################
        # 原始变量赋值保存
        self._items_list_ori = items_list
        self._items_list_selected_ori = items_list_selected
        self._items_list_selected_ascending_ori = items_list_selected_ascending
        
        ####################################################
        # 变量初始化
        self.initial_VARIABLE()
        
        ####################################################
        # widget初始化
        self.initial_WIDGET()
        
        ####################################################
        # event初始化
        self.initial_EVENT()
        
        ####################################################
        # main widget
        self.contruct_main_widget()
        
        
    ####################################################
    # 变量初始化
    def initial_VARIABLE(self):
        
        self.V_ITEMS_LIST = [s0 for s0 in self._items_list_ori if s0 not in self._items_list_selected_ori]
        self.V_ITEMS_LIST_SELECTED = self._items_list_selected_ori
        self.V_ITEMS_LIST_SELECTED_ASCENDING = self._items_list_selected_ascending_ori
                
    
    ####################################################
    # widget初始化
    def initial_WIDGET(self):
        
        ##########################
        self.WIDGET_CHOOSE_COLUMN = widgets.Combobox(
            placeholder="Choose Column",
            options=self.V_ITEMS_LIST,
            description="字段",
            ensure_option=True,
            disabled=False,
        )
        
        ##########################
        self.WIDGET_CHOOSE_COLUMN_ASCENDING = widgets.RadioButtons(
            options=["Ascending", "Descending"],
            value="Ascending",
            # description="",
            # style={"description_width": "initial"},
            disabled=False,
        )
        
        ##########################
        self.WIDGET_BUTTON_ADD_COLUMN = widgets.Button(
            description="{}".format("Add"),
            layout={"width": "70px", "height": "25x"},
            disabled=False,
        )
        
        ##########################
        self.WIDGET_SELECTED_LIST = widgets.Box()
        
        ##########################
        self.OUTPUT_WIDGET_SELECTED_LIST = widgets.Output()
        
    
    ####################################################
    def contruct_main_widget(self):
        
        self.WIDGET_DASHBOARD = widgets.Box(
            children=[
                ##########################
                widgets.Box(
                    children=[
                        self.WIDGET_CHOOSE_COLUMN,
                        self.WIDGET_CHOOSE_COLUMN_ASCENDING,
                        self.WIDGET_BUTTON_ADD_COLUMN,
                    ],
                    layout=widgets.Layout(
                        display="flex",
                        flex_flow="row",
                        border="grey 2px solid",
                        width="100%",
                        height="50px",
                        justify_content="flex-start",
                        align_items="center",
                    ),
                ),
                ##########################
                widgets.Box(
                    children=[
                        self.OUTPUT_WIDGET_SELECTED_LIST,
                    ],
                    layout=widgets.Layout(
                        display="flex",
                        flex_flow="column",
                        border="grey 2px solid",
                        width="100%",
                        height_max="60%",
                        justify_content="flex-start",
                        align_items="center",
                    ),
                ),
            ],
            layout=widgets.Layout(
                display="flex",
                flex_flow="column",
                border="grey 2px solid",
                width="800px",
                height_max="1200px",
                justify_content="center",
                align_items="center",
            ),
        )
        
    
    ####################################################
    # event初始化
    def initial_EVENT(self):
        
        ##########################
        self.WIDGET_BUTTON_ADD_COLUMN.on_click(
            callback=self.EVENT_ON_CLICK_OF_BUTTON_ADD,
        )
        
    
    ####################################################
    def EVENT_ON_CLICK_OF_BUTTON_ADD(self, button_self):
        
        if self.WIDGET_CHOOSE_COLUMN.value in self.V_ITEMS_LIST:
            self.V_ITEMS_LIST_SELECTED \
                .append(self.WIDGET_CHOOSE_COLUMN.value)
            self.V_ITEMS_LIST_SELECTED_ASCENDING \
                .append((True if self.WIDGET_CHOOSE_COLUMN_ASCENDING.value=="Ascending" else False))

            # self.V_ITEMS_LIST.remove(self.WIDGET_CHOOSE_COLUMN.value)
            self.V_ITEMS_LIST = [s0 for s0 in self.V_ITEMS_LIST if s0!=self.WIDGET_CHOOSE_COLUMN.value]
            self.WIDGET_CHOOSE_COLUMN.options = self.V_ITEMS_LIST
            self.WIDGET_CHOOSE_COLUMN.value = ""
            self.WIDGET_CHOOSE_COLUMN_ASCENDING.value = "Ascending"
            
            self._GENERATE_WIDGETS()
        
    
    ####################################################
    def EVENT_ON_CLICK_OF_BUTTON_DELETE(self, button_self):
        
        idx = self.V_ITEMS_LIST_SELECTED.index(button_self.column_name)
        
        self.V_ITEMS_LIST = [
            s0 for s0 in self._items_list_ori
            if s0 in [self.V_ITEMS_LIST_SELECTED[idx]]+self.V_ITEMS_LIST
        ]
        self.WIDGET_CHOOSE_COLUMN.options = self.V_ITEMS_LIST
        
        self.V_ITEMS_LIST_SELECTED \
            .remove(self.V_ITEMS_LIST_SELECTED[idx])
        self.V_ITEMS_LIST_SELECTED_ASCENDING \
            .remove(self.V_ITEMS_LIST_SELECTED_ASCENDING[idx])
        
        self._GENERATE_WIDGETS()
        
    
    ####################################################
    def EVENT_ON_VALUE_CHANGE_OF_SELECTED_WIDGET_CHOOSE_COLUMN(self, change):
        
        for _column, _ascending, _button in [s0.children for s0 in self.WIDGET_SELECTED_LIST.children]:
            _column.placeholder = _column.value
            _button.column_name = _column.value
            
        self.V_ITEMS_LIST_SELECTED = \
            [s0.children[0].value for s0 in self.WIDGET_SELECTED_LIST.children]
        self.V_ITEMS_LIST = [s0 for s0 in self._items_list_ori if s0 not in self.V_ITEMS_LIST_SELECTED]
        
    
    ####################################################
    def EVENT_ON_VALUE_CHANGE_OF_SELECTED_WIDGET_CHOOSE_COLUMN_ASCENDING(self, change):
        
        self.V_ITEMS_LIST_SELECTED_ASCENDING = \
            [(True if s0.children[1].value=="Ascending" else False) for s0 in self.WIDGET_SELECTED_LIST.children]
        
    
    ####################################################
    # 生成widgets
    def _GENERATE_WIDGETS(self):
        
        #############################################################################
        self.WIDGET_SELECTED_LIST = widgets.Box(
            children=[
                widgets.Box(
                    children=[
                        widgets.Combobox(
                            value=_column,
                            placeholder=_column,
                            options=[s0 for s0 in self._items_list_ori if s0 in [_column]+self.V_ITEMS_LIST],
                            description="",
                            ensure_option=True,
                            disabled=False,
                        ),
                        widgets.RadioButtons(
                            options=["Ascending", "Descending"],
                            value=("Ascending" if _ascending else "Descending"),
                            # description="",
                            # style={"description_width": "initial"},
                            disabled=False,
                        ),
                        widgets.Button(
                            description="{}".format("Delete"),
                            layout={"width": "70px", "height": "25x"},
                            disabled=False,
                        ),
                    ],
                    layout=widgets.Layout(
                        display="flex",
                        flex_flow="row",
                        width="800px",
                        justify_content="flex-start",
                        align_items="center",
                    ),
                )
                for _column, _ascending in zip(self.V_ITEMS_LIST_SELECTED, self.V_ITEMS_LIST_SELECTED_ASCENDING)
            ],
            layout=widgets.Layout(
                display="flex",
                flex_flow="column",
                width="800px",
                justify_content="flex-start",
                align_items="center",
            ),
        )
        for _idx, (_column, _ascending, _button) in enumerate([s0.children for s0 in self.WIDGET_SELECTED_LIST.children]):
            
            ##########################
            _column.description = str(_idx)
            _column.observe(
                handler=self.EVENT_ON_VALUE_CHANGE_OF_SELECTED_WIDGET_CHOOSE_COLUMN,
                names="value",
            )
            
            ##########################
            _ascending.observe(
                handler=self.EVENT_ON_VALUE_CHANGE_OF_SELECTED_WIDGET_CHOOSE_COLUMN_ASCENDING,
                names="value",
            )
            
            ##########################
            _button.column_name = _column.value
            _button.on_click(
                callback=self.EVENT_ON_CLICK_OF_BUTTON_DELETE,
            )
        
        #############################################################################
        self.OUTPUT_WIDGET_SELECTED_LIST.clear_output()
        with self.OUTPUT_WIDGET_SELECTED_LIST:
            display(self.WIDGET_SELECTED_LIST)
    


    
#############################################################################################
# widget_DataFrameInspector
# 数据（DataFrame）查看器
class widget_DataFrameInspector(object):
    def __init__(
            self,
            df,
            selection_mapping=None,
        ):
        
        ####################################################
        # 原始变量赋值保存
        self._df_ori = df
        self._selection_mapping_ori = selection_mapping
        
        ####################################################
        # 变量初始化
        self.initial_VARIABLE()
        
        ####################################################
        # widget初始化
        self.initial_WIDGET()
        
        ####################################################
        # event初始化
        self.initial_EVENT()
        
        ####################################################
        # main widget
        self.contruct_main_widget()
        
        
    ####################################################
    # 变量初始化
    def initial_VARIABLE(self):
        
        self.df = self._df_ori.reset_index(drop=True)
        self.df_out = self._df_ori.reset_index(drop=True)
        
    
    ####################################################
    # widget初始化
    def initial_WIDGET(self):
        
        ##########################
        self.WIDGET_QUERY_STRING = widgets.Text(
            description="查询语句（DataFrame Query String）",
            style={"description_width": "initial"},
            disabled=False,
            layout=widgets.Layout(width="50%", height="10%"),
        )
        
        ##########################
        self.WIDGET_COLUMN_DISPLAY = widgets.RadioButtons(
            # options=["Base", "Statistic", "All", "Options"],
            # value="Base",
            options=([] if self._selection_mapping_ori is None else list(self._selection_mapping_ori.keys()))+["All", "Options"],
            value=("All" if self._selection_mapping_ori is None else list(self._selection_mapping_ori.keys())[0]),
            description="字段显示选择",
            style={"description_width": "initial"},
            disabled=False,
        )
        
        ##########################
        self.WIDGET_SELECTION = widget_Selection(
            items_list=self.df.columns.tolist(),
            items_list_selected=[],
        )
        ##########################
        self.WIDGET_SELECTION_ACCORDION = widgets.Accordion(
            children=[
                self.WIDGET_SELECTION.WIDGET_DASHBOARD,
            ],
            layout=widgets.Layout(
                display="flex",
                flex_flow="column",
                # border="grey 2px solid",
                # width="750px",
                # height_max="400px",
                justify_content="center",
                align_items="flex-start",
            ),
        )
        self.WIDGET_SELECTION_ACCORDION.set_title(0, "已选字段")
        self.WIDGET_SELECTION_ACCORDION.selected_index = None
        
        ##########################
        self.WIDGET_OrderSetting = widget_OrderSetting(
            items_list=self._df_ori.columns.tolist(),
            items_list_selected=[],
            items_list_selected_ascending=[],
        )
        
        ##########################
        self.WIDGET_OrderSetting_ACCORDION = widgets.Accordion(
            children=[
                self.WIDGET_OrderSetting.WIDGET_DASHBOARD,
            ],
            layout=widgets.Layout(
                display="flex",
                flex_flow="column",
                # border="grey 2px solid",
                # width="750px",
                # height_max="400px",
                justify_content="center",
                align_items="flex-start",
            ),
        )
        self.WIDGET_OrderSetting_ACCORDION.set_title(0, "字段排序")
        self.WIDGET_OrderSetting_ACCORDION.selected_index = None
        
        ##########################
        self.WIDGET_PageController = widget_PageController(
            df=self.df_out,
            page_no_current=0,
            page_size=20,
            display=True,
        )
        
        ##########################
        self.WIDGET_BUTTON_APPLY = widgets.Button(description="Apply")
        
        ##########################
        self.OUTPUT_DF = widgets.Output()
        # self.OUTPUT_DF = self.WIDGET_PageController.OUTPUT_DF
        
    
    ####################################################
    def contruct_main_widget(self):
        self.WIDGET_DASHBOARD = widgets.Box(
            children=[
                widgets.Box(
                    children=[
                        self.WIDGET_QUERY_STRING,
                        self.WIDGET_COLUMN_DISPLAY,
                        self.WIDGET_SELECTION_ACCORDION,
                        self.WIDGET_OrderSetting_ACCORDION,
                    ],
                    layout=widgets.Layout(
                        display="flex",
                        flex_flow="column",
                        border="grey 2px solid",
                        width="100%",
                        height="20%",
                        justify_content="space-around",
                        # align_content="flex-end",
                    ),
                ),
                self.WIDGET_BUTTON_APPLY,

                widgets.Box(
                    children=[
                        self.OUTPUT_DF,
                    ],
                    layout=widgets.Layout(
                        display="flex",
                        flex_flow="column",
                        border="grey 2px solid",
                        width="100%",
                        height="60%",
                        # height_min="40%",
                        # height_max="100%",
                        # justify_content="space-around",
                        # overflow="scroll",
                    ),
                ),
            ],
            layout=widgets.Layout(
                display="flex",
                flex_flow="column",
                border="grey 4px solid",
                width="1000px",
                # height="1000px",
                height_max="1200px",
                # overflow="scroll",
                justify_content="space-around",
            ),
        )
        
    
    ####################################################
    # event初始化
    def initial_EVENT(self):
        
        ##########################
        # WIDGET_COLUMN_DISPLAY
        self.WIDGET_COLUMN_DISPLAY.observe(
            handler=self.EVENT_ON_VALUE_CHANGE_OF_SELECTED_WIDGET_COLUMN_DISPLAY,
            names="value",
        )
        
        ##########################
        # WIDGET_SELECTION.WIDGET_SELECTMULTIPLE_RIGHT
        self.WIDGET_SELECTION.WIDGET_SELECTMULTIPLE_RIGHT.observe(
            handler=self.EVENT_ON_VALUE_CHANGE_OF_WIDGET_SELECTMULTIPLE_RIGHT,
            names="options",
        )
        
        ##########################
        # WIDGET_SELECTION.WIDGET_BUTTON_SAVE
        self.WIDGET_SELECTION.WIDGET_BUTTON_SAVE.on_click(
            callback=self.EVENT_ON_CLICK_OF_BUTTON_SAVE,
        )
        self.EVENT_ON_VALUE_CHANGE_OF_SELECTED_WIDGET_COLUMN_DISPLAY(change=None)
        
        ##########################
        # WIDGET_BUTTON_APPLY
        self.WIDGET_BUTTON_APPLY.on_click(
            callback=self.EVENT_ON_CLICK_OF_BUTTON_APPLY,
        )
        
    
    ####################################################
    def EVENT_ON_VALUE_CHANGE_OF_SELECTED_WIDGET_COLUMN_DISPLAY(self, change):
        
        if self.WIDGET_COLUMN_DISPLAY.value!="Options":
            
#             self.WIDGET_COLUMN_DISPLAY.columns = (
#                 [
#                     "column_name", "data_type",
#                     "count", "pct_missing", "unique_count", "unique_pct",
#                 ] if self.WIDGET_COLUMN_DISPLAY.value=="Base" else
#                 [
#                     "column_name", "data_type",
#                     "pct_missing", "unique_count",
#                     "min", "mean", "max",
#                     "percentile_05", "percentile_25", "percentile_50", "percentile_75", "percentile_95",
#                     "top5_value", "entropy", "entropy_ratio",
#                 ] if self.WIDGET_COLUMN_DISPLAY.value=="Statistic" else
#                 self.df.columns.tolist() if self.WIDGET_COLUMN_DISPLAY.value=="All" else
#                 []
#             )
            
            if self.WIDGET_COLUMN_DISPLAY.value=="All":
                self.WIDGET_COLUMN_DISPLAY.columns = tuple(self.df.columns.tolist())
            else:
                self.WIDGET_COLUMN_DISPLAY.columns = tuple(self._selection_mapping_ori.get(self.WIDGET_COLUMN_DISPLAY.value, []))
                
            
            self.WIDGET_SELECTION.WIDGET_SELECTMULTIPLE_RIGHT.options = self.WIDGET_COLUMN_DISPLAY.columns
            self.WIDGET_SELECTION.WIDGET_SELECTMULTIPLE_LEFT.options = [s0 for s0 in self.df.columns if s0 not in self.WIDGET_COLUMN_DISPLAY.columns]
            self._WIDGET_COLUMN_DISPLAY_change = 0
            
            self.WIDGET_SELECTION.WIDGET_BUTTON_SAVE.click()
        
    
    ####################################################
    def EVENT_ON_VALUE_CHANGE_OF_WIDGET_SELECTMULTIPLE_RIGHT(self, change):
        
        self._WIDGET_COLUMN_DISPLAY_change = 1
        
#         if len(change["old"])!=len(change["new"]):
#             self._WIDGET_COLUMN_DISPLAY_change = 1
#         elif not all(
#                 np.sort(list(change["old"]))==np.sort(list(change["new"]))
#             ):
#             self._WIDGET_COLUMN_DISPLAY_change = 1
        
    
    ####################################################
    def EVENT_ON_CLICK_OF_BUTTON_SAVE(self, action=None):
        
        ##########################
        self.WIDGET_SELECTION.EVENT_ON_CLICK_OF_BUTTON(action="Save")
        if self._WIDGET_COLUMN_DISPLAY_change==1:
            self.WIDGET_COLUMN_DISPLAY.value = "Options"
        
        ##########################
        self.WIDGET_OrderSetting.V_ITEMS_LIST = self.WIDGET_SELECTION.V_ITEMS_LIST_SELECTED
        self.WIDGET_OrderSetting.V_ITEMS_LIST_SELECTED = [
            s0 for s0 in self.WIDGET_OrderSetting.V_ITEMS_LIST_SELECTED
            if s0 in self.WIDGET_SELECTION.V_ITEMS_LIST_SELECTED
        ]
        self.WIDGET_OrderSetting.V_ITEMS_LIST_SELECTED_ASCENDING = [
            t1 for t0, t1 in
                zip(
                    self.WIDGET_OrderSetting.V_ITEMS_LIST_SELECTED,
                    self.WIDGET_OrderSetting.V_ITEMS_LIST_SELECTED_ASCENDING,
                )
            if t0 in self.WIDGET_SELECTION.V_ITEMS_LIST_SELECTED
        ]
        ##########################
        self.WIDGET_OrderSetting.WIDGET_CHOOSE_COLUMN.options = self.WIDGET_OrderSetting.V_ITEMS_LIST
        self.WIDGET_OrderSetting._GENERATE_WIDGETS()
        
    
    ####################################################
    def EVENT_ON_CLICK_OF_BUTTON_APPLY(self, action):
        
        self.df_out = self.df
        
        ####################################################
        # WIDGET_QUERY_STRING
        if self.WIDGET_QUERY_STRING.value!=None and len(self.WIDGET_QUERY_STRING.value)!=0:
            self.df_out = self.df_out.query(self.WIDGET_QUERY_STRING.value)
        
        ####################################################
        # WIDGET_SELECTION.V_ITEMS_LIST_SELECTED
        if len(self.WIDGET_SELECTION.V_ITEMS_LIST_SELECTED)!=0:
            self.df_out = self.df_out[self.WIDGET_SELECTION.V_ITEMS_LIST_SELECTED]
        
        ####################################################
        # WIDGET_OrderSetting
        if len(self.WIDGET_OrderSetting.V_ITEMS_LIST_SELECTED)!=0:
            self.df_out = self.df_out.sort_values(
                by=self.WIDGET_OrderSetting.V_ITEMS_LIST_SELECTED,
                ascending=self.WIDGET_OrderSetting.V_ITEMS_LIST_SELECTED_ASCENDING,
            ).reset_index(drop=True)
        
        ##########################
        # WIDGET_PageController
        self.WIDGET_PageController = widget_PageController(
            df=self.df_out,
            page_no_current=0,
            page_size=self.WIDGET_PageController.V_PAGE_SIZE,
            display=True,
        )
        
        ####################################################
        self.OUTPUT_DF.clear_output()
        with self.OUTPUT_DF:
            display(self.WIDGET_PageController.WIDGET_DASHBOARD)
            self.WIDGET_PageController.WIDGET_BUTTON_APPLY.click()
        
    
####################################################
if __name__=="__main__":
    pass
    