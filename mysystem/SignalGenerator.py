'''
@ Author: lxxu
@ Date: 2023-12-02 22:04:00
@ Description: 信号生成和处理
@ Email: lixinxu@stu.pku.edu.cn
'''

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")

class SignalGenerator():
    '''
    信号生成
    '''
    def __init__(self,startdate:str='2020-01-01',enddate:str='2022-12-31') -> None:
        PATH=os.path.dirname(os.getcwd())  # 获取当前路径的上一级路径
        # 查看同一目录下是否有data文件夹
        if 'data' not in os.listdir(PATH):
            raise ValueError('File not found! Please put data filefolder in the same path!')
        # 查看data文件夹中是否有数据
        data_list = os.listdir(PATH+'/data')
        if len(data_list)==0:
            raise ValueError('File not found! Please put data filefolder in the same path!')
        
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
        self.stockpool = list(set(price_volume).union(set(income)).union(set(balance)).union(set(cashflow)))

        # 生成交易日列表（量价数据date）
        self.tradedate = self.pricevolume.index.drop_duplicates().tolist()


    def momentum(self,interval:str='5D',is_tradingdate:bool=True)->pd.DataFrame:
        """
        计算动量因子（收盘价）
        :param interval: 计算动量因子的时间间隔
        :param is_tradingdate: 是否按照交易日取天数
        :return momentum: 动量因子
        :example:
            interval = '1M',is_tradingdate = False  计算自然日下1个月的动量因子
            interval = '20D',is_tradingdate = True  计算20个交易日的动量因子
        """
        if interval[-1] not in ['D','M']:  # 判断输入的时间间隔是否正确
            raise ValueError('Interval should end with M or D!')
        if interval[-1]=='M' and is_tradingdate:  # 判断输入的时间间隔和是否按交易日取天数是否匹配
            raise ValueError('Tradingdate should only be counted in D!')
        closeprice=self.pricevolume.pivot_table(index='date',columns='stk_id',values='close')
        if is_tradingdate:   # 按交易日取时间
            factor = closeprice.pct_change(int(interval[:-1]))
        else:   # 按自然日取时间
            if interval[-1]=='M':
                factor = closeprice.pct_change(periods=int(interval[:-1]),freq=pd.DateOffset(months=1))
            else:
                factor = closeprice.pct_change(periods=int(interval[:-1]),freq=pd.DateOffset(days=1))
        factor = factor.reindex(self.tradedate).T.reindex(self.stockpool).T.fillna(method='ffill')
        return factor


class SignalProcessor():
    '''
    信号处理
    '''
    def __init__(self) -> None:
        pass

    
    # 市值中性化
    def MV_Neutral(self, signal:pd.DataFrame)->pd.DataFrame:
        '''
        市值中性化
        :param signal: 信号数据
        :return: 中性化后的数据
        '''
        startdate = signal.index[0]
        enddate = signal.index[-1]
        stock_list_0 = signal.columns.tolist()
        # 读取总市值数据
        path = os.getcwd()
        tot_mv = pd.read_feather(path+'/newdata/tot_mv.feather').loc[startdate:enddate]
        stock_list = list(set(stock_list_0).intersection(set(tot_mv.columns.tolist())))
        tot_mv = tot_mv[stock_list]
        signal = signal[stock_list]
        # 市值中性化数据：每日的signal对于市值进行回归保留残差
        data = signal.copy()
        # 创建一个空的DataFrame
        neutralized_data = pd.DataFrame(index=data.index, columns=data.columns)
        # 遍历每个日期，执行市值中性化回归并计算残差
        for date in data.index:
            try:
                signal = data.loc[date].T
                mv = tot_mv.loc[date].T
                # 去除NaN值
                valid_data = pd.concat([signal, mv], axis=1).dropna()
                valid_data.columns = ['signal', 'mv']
                # 设置信号和总市值作为自变量
                X = sm.add_constant(valid_data['mv'])
                y = valid_data['signal']
                
                # 进行线性回归
                model = sm.OLS(y, X.astype(float)).fit()
                
                # 计算残差并存储在新的DataFrame中
                residuals = model.resid
                neutralized_data.loc[date] = residuals.T
            except:
                pass

        return neutralized_data.T.reindex(stock_list_0).T
    

    # 行业中性化
    def Ind_Neutral(self, signal:pd.DataFrame)->pd.DataFrame:
        '''
        行业中性化
        :param signal: 信号数据
        :return: 中性化后的数据
        '''
        # 读取行业数据
        path = os.getcwd()
        ind_info = pd.read_feather(path+'/newdata/stock_basic.feather')[['ts_code','industry']].set_index('ts_code').T
        ind = ind_info.T
        # 市值中性化数据：每日的signal对于市值进行回归保留残差
        data = signal.copy()
        # 创建一个空的DataFrame
        neutralized_data = pd.DataFrame(index=data.index, columns=data.columns)
        # 遍历每个日期，执行市值中性化回归并计算残差
        for date in data.index:
            try:
                signal = data.loc[date].T
                # 去除NaN值
                valid_data = pd.concat([signal, ind], axis=1).dropna()
                valid_data.columns = ['signal', 'ind']
                # 设置信号和总市值作为自变量
                X = sm.add_constant(pd.get_dummies(valid_data['ind']))
                y = valid_data['signal']
                
                # 进行线性回归
                model = sm.OLS(y, X.astype(float)).fit()
                
                # 计算残差并存储在新的DataFrame中
                residuals = model.resid
                neutralized_data.loc[date] = residuals.T
            except:
                pass

        return neutralized_data
    

    # 中位数去极值
    def MAD_Std(self, signal:pd.DataFrame, limit=5)->pd.DataFrame:
        '''
        中位数去极值
        :param signal: 信号数据
        :param limit: 异常值定义为不在[中位数-limit*MAD,中位数+limit*MAD]范围内的数据
        :return: 处理后的数据
        '''
        data=signal.copy()
        # 计算上下限：中位数+-5*MAD
        med = np.nanmedian(data, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(data - med), axis=1, keepdims=True)
        ub = med+limit*mad
        lb = med-limit*mad
        # 超过上下限则设为上下限
        data1 = data.values
        for dt in range(data1.shape[0]):
            if mad[dt] != 0:
                data1[dt, data1[dt, :] > ub[dt]] = ub[dt]
                data1[dt, data1[dt, :] < lb[dt]] = lb[dt]
        return pd.DataFrame(data1, index=signal.index, columns=signal.columns)
    
    # Rank标准化
    def Rank_Std(self, signal:pd.DataFrame,ascending=False)->pd.DataFrame:
        '''
        对每日信号使用Rank标准化
        :param signal: 信号数据
        :return: 标准化后的数据
        '''
        # 对每行用rank代替原始值
        rank_data = signal.rank(method='average', ascending=ascending, axis=1)
        return rank_data
    
    # Z-score标准化
    def ZScore_Std(self, signal:pd.DataFrame)->pd.DataFrame:
        '''
        对每日信号使用Z-score标准化
        :param signal: 信号数据
        :return: 标准化后的数据
        '''
        data = signal.copy()
        # Z-score标准化每日信号
        Z_data = data.sub(data.mean(axis=1), axis=0).div(data.std(axis=1), axis=0)
        # 若标准差为nan, 则保留原数据
        Z_data.loc[data.std(axis=1).isnull(), :] = data.loc[data.std(axis=1).isnull(), :]

        return Z_data  