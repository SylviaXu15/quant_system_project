'''
@ Author: lxxu
@ Date: 2023-12-01 21:00:00
@ Description: 数据读取和处理
@ Email: lixinxu@stu.pku.edu.cn
'''

import os
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

class DataLoader():
    """ 
    数据读取
    """
    def __init__(self,startdate='2020-01-01',enddate='2022-12-31'):
        PATH=os.path.dirname(os.getcwd())  # 获取当前路径的上一级路径
        
        # 查看同一目录下是否有data文件夹
        if 'data' not in os.listdir(PATH):
            raise ValueError('File not found! Please put data filefolder in the same path!')
        # 查看data文件夹中是否有数据
        data_list = os.listdir(PATH+'/data')
        if len(data_list)==0:
            raise ValueError('File not found! Please put data filefolder in the same path!')
        
        self.path = PATH  # 保存路径
        self.startdate = startdate   # 开始日期
        self.enddate = enddate   # 结束日期
        
        self.pricevolume = pd.read_feather(PATH+'/data/stk_daily.feather')
        self.pricevolume = self.pricevolume[(self.pricevolume['date']>=self.startdate)&(self.pricevolume['date']<=self.enddate)]
        self.pricevolume = self.pricevolume.set_index('date')   # 读取量价数据

        self.balance = pd.read_feather(PATH+'/data/stk_fin_balance.feather')
        self.balance = self.balance[(self.balance['publish_date']>=self.startdate)&(self.balance['publish_date']<=self.enddate)]
        self.balance = self.balance.set_index('publish_date')   # 读取资产负债表
        self.cashflow = pd.read_feather(PATH+'/data/stk_fin_cashflow.feather')
        self.cashflow = self.cashflow[(self.cashflow['publish_date']>=self.startdate)&(self.cashflow['publish_date']<=self.enddate)]
        self.cashflow = self.cashflow.set_index('publish_date')   # 读取现金流量表
        self.income = pd.read_feather(PATH+'/data/stk_fin_income.feather')
        self.income = self.income[(self.income['publish_date']>=self.startdate)&(self.income['publish_date']<=self.enddate)]
        self.income = self.income.set_index('publish_date')   # 读取利润表
        
        # 取量化数据和基本面数据股票的交集为股票池
        price_volume = self.pricevolume['stk_id'].unique().tolist()
        income = self.income['stk_id'].unique().tolist()
        balance = self.balance['stk_id'].unique().tolist()
        cashflow = self.cashflow['stk_id'].unique().tolist()
        # 取并集为股票池
        self.stockpool = sorted(list(set(price_volume).union(set(income)).union(set(balance)).union(set(cashflow))))

        # 生成交易日列表（量价数据date）
        self.tradedate = self.pricevolume.index.drop_duplicates().tolist()


    def datafile_list(self)->list:
        """
        查看data文件夹的data目录
        Note: 用户端文件应当和data文件在同一个目录下
        :return: data目录
        """
        try:
            data_list=os.listdir(self.path+'/data')   # 获取data文件夹下的所有文件名
        except:
            raise ValueError('File not found! Please put data filefolder in the same path!')   # 文件不存在        
        return data_list   # 返回data目录


    def get_fundamental_dict(self)->dict:
        """
        查看基本面数据字典
        :param printout: 是否打印基本面数据列表
        :return: 基本面数据列表
        """
        try:
            item_map = pd.read_feather(self.path+'/data/stk_fin_item_map.feather')  # 读取基本面数据的指标映射表
            item_dict = {}   # 初始化指标字典
            for table in item_map['table'].unique().tolist():   # 遍历三张财报
                item_dict[table] = item_map[item_map['table']==table]['item'].unique().tolist()   # 生成三张财报的指标字典
        except:
            raise ValueError('File not found!')   # 文件不存在
        return item_dict


    def load_pricevol_data(self,genre:str='close',adjusted:bool=True)->pd.DataFrame:
        """
        读取量价数据
        :param genre: 数据类型  ['open','high','low','close','volume','amount']
        :param adjusted: 是否复权(仅针对价格数据)
        :return: 数据
        """
        PRICE_VOLUME = ['open','high','low','close','volume','amount']   # 量价数据
        if genre not in PRICE_VOLUME:   # 判断输入的数据类型是否正确
            raise ValueError('Genre should be in '+','.join(PRICE_VOLUME))
        file = pd.read_feather(self.path+'/data/stk_daily.feather')   # 读取量价数据
        data = file.pivot_table(index='date',columns='stk_id',values=genre)   # 生成期望数据的透视表
        if adjusted and genre in ['open','high','low','close']:   # 判断是否复权
            adjfactor = file.pivot_table(index='date',columns='stk_id',values='cumadj')   # 生成复权因子的透视表
            data = data*adjfactor   # 价格数据乘以复权因子
        data = data.reindex(index=self.tradedate,columns=self.stockpool)   # 重新索引
        return data   # 返回数据
    

    def load_fundamental_data(self,genre:str,fill:bool=True)->pd.DataFrame:
        """
        读取基本面数据
        :param genre: 数据类型  
        :param fill: 是否填充缺失值 默认ffill
        :return: 数据
        """ 
        try:
            item_map = pd.read_feather(self.path+'/data/stk_fin_item_map.feather')  # 读取基本面数据的指标映射表
            target = item_map[item_map['item']==genre]   # 获取目标数据的指标映射
            table = target['table'].values[0]   # 获取目标数据的财报
            field = target['field'].values[0]   # 获取目标数据的指标
            file = pd.read_feather(self.path+'/data/stk_fin_'+table+'.feather')   # 读取目标数据
            file = file[(self.startdate<=file['publish_date']) &(file['publish_date']<=self.enddate)]   # 筛选开始日期
            data = file.pivot_table(index='publish_date',columns='stk_id',values=field)   # 生成期望数据的透视表
            data = data.reindex(index=self.tradedate,columns=self.stockpool)   # 重新索引
            if fill:   # 判断是否填充缺失值
                data = data.fillna(method='ffill')   # 填充缺失值
        except:
            raise ValueError('Error!')   # 出现错误
        return data   # 返回数据
    

    def load_newdata(self,genre:str)->pd.DataFrame:
        """
        读取newdata文件夹数据
        :param genre: 数据类型 默认包括[tot_mv,index_basic,index_close,stock_basic]  
        :return: 数据
        """ 
        # 获取当前路径
        cur_path =os.getcwd()
        newdata_list = os.listdir(cur_path+'/newdata')   # 获取newdata文件夹下的所有文件名
        if genre+'.feather' not in newdata_list:   # 判断输入的数据类型是否正确
            raise ValueError('Genre should be in '+','.join(newdata_list))
        data = pd.read_feather(cur_path+'/newdata/'+genre+'.feather')   # 读取目标数据
        return data