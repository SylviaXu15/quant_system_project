'''
@ Author: lxxu
@ Date: 2023-12-06 15:21:00
@ Description: 回测
@ Email: lixinxu@stu.pku.edu.cn
'''

import numpy as np
import pandas as pd
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 把当前文件添加到搜寻路径，以导入同级py文件

import DataLoader   # 导入自定义库：数据读取器
import Presentation   # 导入自定义库：数据展示器


class SignalBacktest():
    '''
    对信号进行回测
    : Step 1: __init__() 初始化回测对象以及回测区间
    : Step 2: load_signal() 读取回测信号
    : Step 3: strategy_eval()  进行单策略回测
    '''
    def __init__(self, startdate:str='2020-01-01',enddate:str='2022-12-31')->None:
        '''
        初始化
        : param startdate: 回测开始日期 str eg.'2020-01-01'
        : param enddate: 回测结束日期 str eg.'2022-12-31'
        '''
        self.startdate = startdate
        self.enddate = enddate

        dataloader = DataLoader.DataLoader(self.startdate,self.enddate)   # 初始化数据读取器对象
        self.stock_close = dataloader.load_pricevol_data('close',False)   # 未复权收盘价
        self.stock_close_adj = dataloader.load_pricevol_data('close',True)   # 复权收盘价
        self.tradedates = dataloader.tradedate   # 交易日列表
        path = os.getcwd()
        self.index_close_agg = pd.read_feather(path+'/newdata/index_close.feather')  # 读取市场指数收盘价
        self.index_close = self.index_close_agg['000001.SH']   # 默认基准为上证指数
        self.testtools = BacktestTools()   # 初始化回测工具
        self.presenter = Presentation.Presenter()   # 创建结果展示器实例
        print('Please use the load_signal() method to load the signal!')   # 提示用户使用load_signal()方法读取信号


    def load_signal(self, signal_type:str='df',signal_file:str='./',signal_df:pd.DataFrame=pd.DataFrame())->None:
        '''
        读取信号
        : param signal_type: 信号类型，'path'为路径，'df'为DataFrame
        : param signal_path: 信号路径  eg.'./signal.feather'
        : param signal_df: 信号DataFrame
        '''
        if signal_type=='path':   # 信号通过路径加载
            try:
                self.signal = pd.read_feather(signal_file)   # 读取信号
            except:   # 读取失败
                raise ValueError('File not found! Please enter a correct path!')
        elif signal_type=='df':   # 信号通过DataFrame加载
            self.signal = signal_df   # 读取信号
        else:   # 信号类型错误
            raise ValueError('signal_type must be "path" or "df"!')
        
    def change_base(self, base:str):
        '''
        修改回测设定的基准
        '''
        try:
            self.index_close = self.index_close_agg[base]
        except:
            raise KeyError("please enter an index in the index list.\n\
                           To query the index list you can use the load_newdata api in DataLoader")


    def strategy_eval(self):
        """
        对单个策略进行回测
        :return: 回测结果 
            : 0. stats_results [1*10]: IC/rank_IC均值/标准差,IC_IR,|t|均值,|t|>2占比,t均值,因子收益率均值,R方均值
            : 1. stats_series [T*7]: (累计)IC, (累计)rank_IC, t, 因子收益率, R方时间序列
            : 2. clsfy_nav [T*11]: 分层1~5净值, 基准净值, 分层1~5相对净值序列
            : 3. clsfy_perf [6*10]: 分层1~5年化收益率,年化波动率,夏普比率,最大回撤,年化超额收益率,超额收益年化波动率,
                                    信息比率,超额收益最大回撤,调仓胜率,相对基准盈亏比
            : 4. ls_nav [T*8]: 多头, 空头, 多空, 基准, 多头相对净值, 空头相对净值, 多空相对净值序列
            : 5. ls_perf [4*10]: 多头/空头/多空年化收益率,年化波动率,夏普比率,最大回撤,年化超额收益率,超额收益年化波动率,
                                    信息比率,超额收益最大回撤,调仓胜率,相对基准盈亏比
        """
        signal = self.signal.copy()   # 复制信号，防止修改原信号

        # 去掉全空的行列，用nan替换inf
        signal = signal.dropna(how='all',axis=1)
        signal = signal.dropna(how='all',axis=0)
        signal = signal.replace([np.inf,-np.inf],np.nan)

        # 统计性指标计算（默认月频）
        stats_results, stats_series = self.stats_test(signal, freq='M')
        
        # 分层回测（默认季频）
        clsfy_nav, clsfy_port, clsfy_perf = self.clsfy_backtest(signal,
                                                                freq='Q',
                                                                start_date=signal.index[0],
                                                                end_date=signal.index[-1],
                                                                layer_number=5)

        # 多空回测（默认季频）
        ls_nav, ls_port, ls_perf = self.ls_backtest(signal,
                                                    freq='Q',
                                                    start_date=signal.index[0],
                                                    end_date=signal.index[-1],
                                                    quantile=0.1)
        
        # 保存结果
        self.results = {}
        self.results['signal'] = self.signal
        self.results['stats_results'] = stats_results
        self.results['stats_series'] = stats_series
        self.results['clsfy_nav'] = clsfy_nav
        self.results['clsfy_perf'] = clsfy_perf
        self.results['ls_nav'] = ls_nav
        self.results['ls_perf'] = ls_perf

        # 返回统计性指标，分层回测结果，多空回测结果
        return self.results
    

    def stats_test(self, signal:pd.DataFrame, freq:str)->(pd.DataFrame,pd.DataFrame):
        '''
        计算统计性指标
        :param signal: 信号
        :param freq: 重采样频率
        :return: 统计性指标结果 IC,回归
            : 0. stats_result: IC/rank_IC均值/标准差,IC_IR,|t|均值,|t|>2占比,t均值,因子收益率均值,R方均值
            : 1. stats_series: (累计)IC, (累计)rank_IC, t, 因子收益率, R方时间序列
        '''
        signal = signal.copy()   # 复制信号，防止修改原信号

        signal.index = pd.to_datetime(signal.index)   # 转换为时间序列
        signal = signal.resample(freq).last()   # 按照频率重采样

        if signal.index[-1] >= self.tradedates[-1]:   # 最后一个交易信号无法调仓
            signal = signal.drop(index=signal.index[-1])   # 删除最后一个交易信号
        
        # 标准化每日信号分布
        signal = signal.sub(signal.mean(axis=1),axis=0).div(signal.std(axis=1),axis=0)
        
        # 生成调仓日期：信号日期的下一交易日
        tradedates = pd.Series(data=self.tradedates, index=self.tradedates)
        refresh_dates = [tradedates[tradedates > i].index[0] for i in signal.index]

        # 计算股票下期收益率
        stock_close = self.stock_close_adj.loc[refresh_dates, signal.columns]
        stock_return_next = stock_close.pct_change().replace(np.inf,np.nan).shift(-1)
        stock_return_next.index = signal.index   # 修改日期索引

        # ====================================================
        # IC系列指标计算
        # ====================================================
        # 计算IC和rank_IC
        IC = signal.corrwith(stock_return_next, method='pearson', axis=1)
        rank_IC =signal.corrwith(stock_return_next, method='spearman', axis=1)
        IC_IR = IC.mean() / IC.std()

        # 累计IC和rank_IC
        IC_cum = IC.fillna(0).cumsum()
        rank_IC_cum = rank_IC.fillna(0).cumsum()

        # 汇总IC结果
        IC_results = pd.concat([IC, rank_IC, IC_cum, rank_IC_cum], axis=1)
        IC_results.columns = ['IC', 'rank_IC', '累计IC', '累计rank_IC']

        # ====================================================
        # 线性回归测试
        # ====================================================
        signal_trans = signal.T
        stock_return_trans = stock_return_next.T
        linear_reg_results = pd.DataFrame(index=signal.index, columns=['t值', '因子收益率', 'R方'])
        for date in signal.index:
            # X和Y均不全为nan值
            if ~signal_trans[date].isnull().all() and ~stock_return_trans[date].isnull().all():
                x = sm.add_constant(signal_trans[date])   # 加入截距项
                y = stock_return_trans[date].replace(np.inf,np.nan)   # 替换inf为nan
                reg_result = sm.OLS(y, x, missing='drop').fit()   # 回归
                # 存放回归结果
                linear_reg_results.loc[date, 't值'] = reg_result.tvalues[1]
                linear_reg_results.loc[date, '因子收益率'] = reg_result.params[1]
                linear_reg_results.loc[date, 'R方'] = reg_result.rsquared
        # 汇总结果
        stats_result = pd.DataFrame(index=['值'], columns=['IC均值', 'IC标准差', 'rank_IC均值', 'rank_IC标准差', 'IC_IR',
                                                          '|t|均值', '|t|>2占比', 't均值', '因子收益率均值', 'R方均值'])
        stats_result.loc['值', 'IC均值'] = IC.mean()
        stats_result.loc['值', 'IC标准差'] = IC.std()
        stats_result.loc['值', 'rank_IC均值'] = rank_IC.mean()
        stats_result.loc['值', 'rank_IC标准差'] = rank_IC.std()
        stats_result.loc['值', 'IC_IR'] = IC_IR

        stats_result.loc['值', '|t|均值'] = linear_reg_results['t值'].abs().mean()
        stats_result.loc['值', '|t|>2占比'] = (linear_reg_results['t值'].abs() > 2).sum() / linear_reg_results.shape[0]
        stats_result.loc['值', 't均值'] = linear_reg_results['t值'].mean()
        stats_result.loc['值', '因子收益率均值'] = linear_reg_results['因子收益率'].mean()
        stats_result.loc['值', 'R方均值'] = linear_reg_results['R方'].mean()
        # 序列结果
        stats_series = pd.concat([IC_results, linear_reg_results], axis=1)

        return stats_result, stats_series


    def clsfy_backtest(self,signal:pd.DataFrame=pd.DataFrame(),freq:str='Q',start_date:str='',end_date:str='',layer_number:int=5,fee:float=0)->(pd.DataFrame,pd.DataFrame,pd.DataFrame):
        '''
        分层回测
        :param signal: 信号
        :param freq: 重采样频率
        :param start_date: 回测开始日期
        :param end_date: 回测结束日期
        :param layer_number: 分层数量
        :param fee: 手续费
        :return: 回测结果
            : 0. nav: 各分层净值, 基准净值, 各分层相对净值序列
            : 1. port: 各分层持仓权重序列 list[pd.DataFrame]
            : 2. performance: 分层1~5年化收益率,年化波动率,夏普比率,最大回撤,年化超额收益率,超额收益年化波动率,
                              信息比率,超额收益最大回撤,调仓胜率,相对基准盈亏比
        '''
        # 初始化参数
        if len(signal)==0:   # 未输入信号
            signal = self.signal.copy()
        if start_date=='':   # 未输入开始日期
            start_date = self.startdate
        if end_date=='':   # 未输入结束日期
            end_date = self.enddate

        # 回测净值
        nav = pd.DataFrame(columns=['分层1', '分层2', '分层3', '分层4', '分层5', '基准',
                                    '分层1相对净值', '分层2相对净值', '分层3相对净值', '分层4相对净值', '分层5相对净值'])

        # 参照基准：市场指数净值
        base_close = self.index_close.loc[start_date:end_date]
        base_nav = base_close / base_close.iloc[0]
        
        monthly_signal = signal.resample(freq).last().copy()   # 重采样
        if monthly_signal.index[-1] >= self.tradedates[-1]:   # 最后一个交易信号无法调仓
            monthly_signal = monthly_signal.drop(index=monthly_signal.index[-1])   # 删除最后一个交易信号

        # 生成调仓日期：信号日期的下一交易日
        tradedates = pd.Series(data=self.tradedates, index=self.tradedates)
        refresh_dates = [tradedates[tradedates > i].index[0] for i in monthly_signal.index]
        monthly_signal.index=refresh_dates   # 修改日期索引

        # 替换为从大到小的排序次序
        signal_rank = monthly_signal.rank(method='average', ascending=False, axis=1)

        # 确定分层股票数和分割点
        layer_stock_num = np.int64(np.floor(signal_rank.shape[1] / layer_number))
        inter_part = signal_rank.shape[1] % layer_number / layer_number
        thres = layer_stock_num + inter_part

        # 各层持仓权重
        port = []

        # 确定各层持仓
        for layer_id in range(layer_number):
            # 本层排名上下限
            thres_up = 1 / layer_number * (layer_id + 1)
            thres_down = 1 / layer_number * layer_id
            # 初始化本层持仓数据
            factor_layer = pd.DataFrame(np.zeros_like(monthly_signal.values), index=monthly_signal.index,
                                        columns=monthly_signal.columns)
            # 属于本层的股票设为1
            factor_layer[(signal_rank.apply(lambda x: x > x.max() * thres_down, axis=1)) &
                         (signal_rank.apply(lambda x: x <= x.max() * thres_up, axis=1))] = 1
            # 全为零行替换为全仓
            factor_layer[(factor_layer == 0).sum(axis=1) == factor_layer.shape[1]] = 1 / factor_layer.shape[1]
            # 空值替换为全仓
            factor_layer[factor_layer.isnull().sum(axis=1) == factor_layer.shape[1]] = 1 / factor_layer.shape[1]
            # 无因子值的股票权重置为0
            factor_layer[signal_rank.isnull()] = 0
            # 持仓归一化
            factor_layer = (factor_layer.multiply(1 / factor_layer.sum(axis=1), axis=0)).loc[start_date:end_date, :]
            port.append(factor_layer)
            # 回测,计算策略净值
            nav[f'分层{5-layer_id}'], _ = self.testtools.cal_nav(factor_layer,
                                                                self.stock_close_adj.loc[start_date:end_date,
                                                                factor_layer.columns],
                                                                base_nav,
                                                                fee=fee)
        # 基准净值归一化
        nav['基准'] = base_nav.reindex(nav.index)
        # 计算相对净值
        for layer_id in range(layer_number):
            nav[f'分层{layer_id+1}相对净值'] = nav[f'分层{layer_id+1}'] / nav['基准']

        # 计算业绩指标
        performance = self.nav_perf(nav.loc[:, '分层1':'基准'], list(port[0].index))

        # 返回绝对净值曲线，相对净值曲线
        return nav, port, performance
    

    def ls_backtest(self, signal:pd.DataFrame=pd.DataFrame(), freq:str='Q', start_date:str='', end_date:str='', quantile:float=0.1,fee:float=0)->(pd.DataFrame,pd.DataFrame,pd.DataFrame):
        '''
        多空回测
        :param signal: 信号
        :param freq: 重采样频率
        :param start_date: 回测开始日期
        :param end_date: 回测结束日期
        :param quantile: 分位数
        :param fee: 手续费
        :return: 回测结果
            : 0. nav: 多头, 空头, 多空, 基准, 多头相对净值, 空头相对净值, 多空相对净值序列
            : 1. port: 多头/空头持仓权重序列 list[pd.DataFrame]
            : 2. performance: 多头/空头/多空年化收益率,年化波动率,夏普比率,最大回撤,年化超额收益率,超额收益年化波动率,
                              信息比率,超额收益最大回撤,调仓胜率,相对基准盈亏比
        '''
        # 初始化参数
        if len(signal)==0:
            signal = self.signal.copy()
        if start_date=='':
            start_date = self.startdate
        if end_date=='':
            end_date = self.enddate

        # 回测净值
        nav = pd.DataFrame(columns=['多头', '空头', '多空', '基准', '多头相对净值', '空头相对净值', '多空相对净值'])

        # 参照基准：市场指数净值
        base_close = self.index_close.loc[start_date:end_date]
        base_nav = base_close / base_close.iloc[0]
        
        monthly_signal = signal.resample(freq).last().copy()   # 重采样
        if monthly_signal.index[-1] >= self.tradedates[-1]:   # 最后一个交易信号无法调仓
            monthly_signal = monthly_signal.drop(index=monthly_signal.index[-1])   # 删除最后一个交易信号

        # 生成调仓日期：信号日期的下一交易日
        tradedates = pd.Series(data=self.tradedates, index=self.tradedates)
        refresh_dates = [tradedates[tradedates > i].index[0] for i in monthly_signal.index]

        # 指标进行排序
        signal_rank = monthly_signal.rank(method='average', ascending=False, axis=1)

        # 各层持仓权重
        port = []

        # 多头持仓
        factor_layer_long = pd.DataFrame(np.zeros_like(monthly_signal.values), index=monthly_signal.index, columns=monthly_signal.columns)
        factor_layer_long[(signal_rank.apply(lambda x: x > 0, axis=1)) &
                          (signal_rank.apply(lambda x: x <= x.max() * quantile, axis=1))] = 1

        # 空头持仓
        factor_layer_short = pd.DataFrame(np.zeros_like(monthly_signal.values), index=monthly_signal.index, columns=monthly_signal.columns)
        factor_layer_short[(signal_rank.apply(lambda x: x > x.max() * (1 - quantile), axis=1)) &
                           (signal_rank.apply(lambda x: x <= x.max(), axis=1))] = 1

        # 全为零行替换为全仓
        factor_layer_long[(factor_layer_long == 0).sum(axis=1) == factor_layer_long.shape[1]] = 1 / factor_layer_long.shape[1]
        factor_layer_short[(factor_layer_short == 0).sum(axis=1) == factor_layer_short.shape[1]] = 1 / factor_layer_short.shape[1]
        # 空值替换为全仓
        factor_layer_long[factor_layer_long.isnull().sum(axis=1) == factor_layer_long.shape[1]] = 1 / factor_layer_long.shape[1]
        factor_layer_short[factor_layer_short.isnull().sum(axis=1) == factor_layer_short.shape[1]] = 1 / factor_layer_short.shape[1]

        # 无因子值的股票权重置为0
        factor_layer_long[signal_rank.isnull()] = 0
        factor_layer_short[signal_rank.isnull()] = 0

        # 持仓归一化
        factor_layer_long = (factor_layer_long.multiply(1 / factor_layer_long.sum(axis=1), axis=0)).loc[start_date:end_date, :].fillna(0)
        factor_layer_short = (factor_layer_short.multiply(1 / factor_layer_short.sum(axis=1), axis=0)).loc[start_date:end_date, :].fillna(0)

        port.append(factor_layer_long)
        port.append(factor_layer_short)

        # 回测, 计算策略净值
        nav['多头'], _ = self.testtools.cal_nav(factor_layer_long,
                                             self.stock_close_adj.loc[start_date:end_date, monthly_signal.columns],
                                             base_nav,
                                             fee=fee)
        nav['空头'], _ = self.testtools.cal_nav(factor_layer_short,
                                             self.stock_close_adj.loc[start_date:end_date, monthly_signal.columns],
                                             base_nav,
                                             fee=fee)
        nav['多空'] = ((nav['多头'].pct_change() - nav['空头'].pct_change()).fillna(0) + 1).cumprod()

        # 基准净值归一化
        nav['基准'] = base_nav.loc[nav.index] / base_nav[nav.index[0]]

        # 计算相对净值
        nav['多头相对净值'] = nav['多头'] / nav['基准']
        nav['空头相对净值'] = nav['空头'] / nav['基准']
        nav['多空相对净值'] = nav['多空'] / nav['基准']

        # 计算业绩指标
        performance = self.nav_perf(nav.loc[:, '多头':'基准'], list(port[0].index))

        # 返回绝对净值曲线，相对净值曲线
        return nav, port, performance


    def nav_perf(self, nav, refresh_dates):
        '''
        根据净值序列计算业绩指标
        :param nav: 净值序列 T*N (T为时间, N为测试的净值序列数量) 默认最后一列为基准净值
        :param refresh_dates: 调仓日期
        :return: 业绩指标
        '''
        if '基准' not in nav.columns:
            raise ValueError('The last column of nav must be the benchmark with the name "基准"!')
        # 计算回测业绩指标
        perf_all = []
        for target in nav.columns[0:-1]:
            perf_all.append(self.testtools.excess_statis(nav[target], nav['基准'], refresh_dates).loc['策略', :])
        perf_all.append(self.testtools.excess_statis(nav[target], nav['基准'], refresh_dates).loc['基准', :])
        # 结果汇总
        merge_perf_all = pd.concat(perf_all, axis=1)
        merge_perf_all.columns = nav.columns
        return merge_perf_all.T
    
    def corr_matrix(self, signal_dict:dict={}, plot:bool=True, save:bool=False, path:str='./'):
        '''
        计算因子相关性矩阵
        :param plot: 是否绘制热力图
        :param signal_dict: 因子字典
        '''
        # 因子相关性
        corr_sheet = pd.DataFrame(index=signal_dict.keys(), columns=signal_dict.keys())
        for key1, value1 in signal_dict.items():
            for key2, value2 in signal_dict.items():
                if key1 == key2:
                    corr_sheet.loc[key1, key2] = 1
                    continue
                if corr_sheet.isnull().loc[key1, key2]:
                    corr_series = value1.corrwith(value2, method='spearman', axis=1)
                    corr_sheet.loc[key1, key2] = corr_series.mean()
                    corr_sheet.loc[key2, key1] = corr_series.mean()
        if plot:
            self.presenter.plot_heatmap(corr_sheet)
        if save:
            corr_sheet.to_excel(path+'因子相关性矩阵.xlsx')

        return corr_sheet


    def strategy_merge(self, signal_dict:dict={}, weight_list:list=[]):
        '''
        简单的策略复合: 按比例复合策略, weight_list为[]则视为等权重
        AI复合策略此系统暂不涉及
        : param signal_dict: 信号字典
        : param weight_list: 复合的权重
        '''
        if len(weight_list)==0:
            weight_list = [1/len(signal_dict)]*len(signal_dict)
        if len(signal_dict)!=len(weight_list):
            raise ValueError('The length of signal_list and weight_list must be equal!')
        # 因子转换频率
        signal_list = [signal.loc[self.startdate:self.enddate, :] for signal in signal_dict.values()]

        # 计算因子在截面上的标准化分数
        signal_zscore = [self.testtools.normalize(self.testtools.med_std(signal)) for signal in signal_list]

        # 将所有因子标准化值(中位数去极值后)加和平均
        zscore_sum = np.ndarray([len(signal_dict), signal_zscore[0].shape[0], signal_zscore[0].shape[1]])
        for i, signal in enumerate(signal_zscore):
            zscore_sum[i, :, :] = weight_list[i]*signal.values
        zscore_sum_mean = np.nansum(zscore_sum, axis=0)/sum(weight_list)
        zscore_sum_mean_df = pd.DataFrame(data=zscore_sum_mean, index=signal_zscore[0].index,
                                          columns=signal_zscore[0].columns)
        # 再次标准化
        signal_composite = zscore_sum_mean_df.sub(zscore_sum_mean_df.mean(axis=1), axis=0).div(
            zscore_sum_mean_df.std(axis=1), axis=0)
        signal_composite = signal_composite.dropna(how='all', axis=0)

        return signal_composite


    def save_results(self, save_as:list[str]=['feather'], save_path:str='./'):
        '''
        保存回测结果到本地
        :param save_as: 保存格式, 可以选择 feather pickle excel, 默认为feather
        :param save_path: 保存路径 默认为当前路径
        '''
        for item in save_as:
            if item=='excel':
                excel_writer = pd.ExcelWriter(save_path+'Backtest_Results.xlsx')
                self.results['stats_results'].to_excel(excel_writer, sheet_name=f'统计检验指标')
                self.results['stats_series'].to_excel(excel_writer, sheet_name=f'统计检验时间序列')
                self.results['clsfy_nav'].to_excel(excel_writer, sheet_name=f'分层回测多期净值')
                self.results['clsfy_perf'].to_excel(excel_writer, sheet_name=f'分层回测回测指标')
                self.results['ls_nav'].to_excel(excel_writer, sheet_name=f'多空回测多期净值')
                self.results['ls_perf'].to_excel(excel_writer, sheet_name=f'多空回测回测指标')
                excel_writer.close()
            elif item=='feather':
                self.results.to_feather(save_path+'Backtest_Results.feather')
            elif item=='pickle':
                self.results.to_pickle(save_path+'Backtest_Results.pickle')
            else:
                raise ValueError('save_as must be "feather" or "pickle" or "excel"!')


    def plot_results(self, save:bool=False, save_path:str='./'):
        '''
        绘制回测结果
        : param save: 是否保存 默认为False
        : param save_path: 保存路径 默认为当前路径
        '''
        self.presenter.save = save
        self.presenter.path = save_path
        self.presenter.load_results(self.results)  # 读取回测结果
        self.presenter.multi_plot()   # 绘制回测结果


    

class BacktestTools():
    def __init__(self) -> None:
        pass

    def cal_nav(self, signal, adjclose, base_nav, fee):  
        '''
        计算净值曲线
        :param signal: 信号
        :param adjclose: 调仓日收盘价
        :param base_nav: 基准净值
        :param fee: 手续费
        :return: 净值曲线, 换手率
        '''   
        # 获取所有调仓日期
        refresh_dates = signal.index.tolist()
        # 节选出回测区间内的收盘价
        adjclose = adjclose.loc[refresh_dates[0]:,:]
        # 获取回测区间日频交易日
        tradedates = adjclose.index.tolist()  
        # 初始化净值曲线
        nav = pd.Series(index=adjclose.index, name='策略', dtype=float)
        # 初始化换手率记录
        turn = pd.Series(index=refresh_dates, name='当期换手', dtype=float)
        # 初始化日期计数器
        date_index = 0  
        
        # 遍历每个日期
        for date_index in range(len(tradedates)):
            # 当日日期
            date = tradedates[date_index]
            
            # 如果是回测期首日：初次建仓
            if date_index == 0:  
                # 获取当前调仓权重
                new_weight = signal.loc[date,:]                 
                # 计算当前持仓个股净值，考虑第一次调仓的手续费
                portfolio = (1-fee)*new_weight
                # 记录净值
                nav[date] = 1-fee
                # 直接进行下一次循环
                continue
            
            # 根据个股涨跌幅更新组合净值，将日期计数器自增1
            # 当天收盘价
            cur_close = adjclose.iloc[date_index, :]
            # 上一天的收盘价
            prev_close = adjclose.iloc[date_index-1, :]
            # 判断最新的收盘价是否存在空值
            cur_close_nan = cur_close[cur_close.isna()].index
            # 当存在持有资产价格为空的情况时，重新计算权重分布，剔除此种资产
            if np.nansum(portfolio[cur_close_nan])> 0:
                # 提取前一个日期
                prev_date = tradedates[date_index-1]
                # 归一化当前持仓中个股权重, 空值记为0
                old_weight = portfolio / np.nansum(np.abs(portfolio))
                old_weight[old_weight.isnull()] = 0
                # 获取最新的持仓权重
                new_weight = old_weight.copy()
                new_weight[cur_close_nan]=0
                # 归一化当前持仓中个股权重, 空值记为0
                new_weight = new_weight / np.nansum(np.abs(new_weight))
                new_weight[new_weight.isnull()] = 0
                # 直接按照新的持仓组合分配权重
                portfolio = new_weight * nav[prev_date]
            # 根据涨跌幅更新组合净值
            portfolio = cur_close / prev_close * portfolio
            # 未持有资产时，组合净值维持不变
            if np.nansum(portfolio) == 0:
                nav[date] = nav.iloc[tradedates.index(date) - 1]
            else:
                nav[date] = np.nansum(portfolio)

            # 如果是调仓日，执行调仓操作
            if date in refresh_dates:
                # 归一化当前持仓中个股权重
                old_weight = portfolio / np.nansum(np.abs(portfolio))
                old_weight[old_weight.isnull()] = 0
                # 获取最新的持仓权重
                new_weight = signal.loc[date,:] 
                # 计算换手率，最小为0，也即不换仓，最大为2，也就是全部换仓
                turn_over = np.sum(np.abs(new_weight - old_weight))
                turn[date] = turn_over / 2
                # 更新换仓后的净值，也即扣除手续费
                nav[date] = nav[date] * (1 - turn_over * fee)
                # 更新持仓组合中个股的最新净值
                portfolio = new_weight * nav[date]
        return nav, turn
    
    
    def excess_statis(self, nav_seq, base_seq, refresh_dates):
        '''
        计算超额收益的业绩指标
        :param nav_seq: 净值序列
        :param base_seq: 基准序列
        :param refresh_dates: 调仓日期
        :return: 业绩指标
        '''
        # 初始化结果矩阵
        perf = pd.DataFrame(index=['策略', '基准'], 
                            columns=['年化收益率','年化波动率','夏普比率','最大回撤',
                                    '年化超额收益率','超额收益年化波动率','信息比率',
                                    '超额收益最大回撤','调仓胜率', '相对基准盈亏比'])
        # 统计策略和基准的基本业绩指标
        perf.iloc[:, :4] = self.normal_statis(pd.concat([nav_seq, base_seq], axis=1)).values
        # 计算策略相比于基准的超额收益
        excess_return = nav_seq.pct_change() - base_seq.pct_change()
        # 计算超额收益率累计净值
        excess_nav = (1 + excess_return.fillna(0)).cumprod()
        # 计算超额收益的业绩指标
        perf.iloc[0, 4:8] = self.normal_statis(excess_nav).values
        # 计算胜率
        perf.loc['策略', '调仓胜率'] = self.win_rate(nav_seq, base_seq, refresh_dates)
        # 计算胜率
        perf.loc['策略', '相对基准盈亏比'] = self.profit_loss_ratio(nav_seq, base_seq, refresh_dates)
        return perf
    
    def normal_statis(self, nav):
        '''
        计算基本业绩指标
        :param nav: 净值序列
        :return: 业绩指标
        '''
        # 获取所有净值曲线的年化收益率与年化波动率
        perf = nav.apply([self.annualized_return, self.annualized_volatility, self.sharp_ratio, self.max_drawdown])
        # 设置指标名称
        perf.index = ['年化收益率', '年化波动率', '夏普比率', '最大回撤']
        return perf.T

    def annualized_return(self, nav):
        '''
        计算年化收益率
        :param nav: 净值序列
        :return: 年化收益率
        '''
        return pow(nav[-1] / nav[0], 250/len(nav)) - 1

    def annualized_volatility(self, nav):
        '''
        计算年化波动率
        :param nav: 净值序列
        :return: 年化波动率
        '''
        return nav.pct_change().std() * np.sqrt(250)

    def sharp_ratio(self, nav):
        '''
        计算夏普比率
        :param nav: 净值序列
        :return: 夏普比率
        '''
        return self.annualized_return(nav) / self.annualized_volatility(nav)
            
    def max_drawdown(self, nav):
        '''
        计算最大回撤
        :param nav: 净值序列
        :return: 最大回撤
        '''
        max_drawdown = 0
        # 遍历每一天
        for index in range(1, len(nav)):
            cur_drawdown = nav[index] / max(nav[0:index]) - 1
            if cur_drawdown < max_drawdown:
                max_drawdown = cur_drawdown
        return max_drawdown
  
    def win_rate(self, strategy_nav, base_nav, refresh_dates):
        '''
        计算调仓胜率
        :param strategy_nav: 策略净值
        :param base_nav: 基准净值
        :param refresh_dates: 调仓日期
        :return: 调仓胜率
        '''
        # 抽取调仓日策略和基准的净值
        resampled_strategy = strategy_nav[refresh_dates].tolist()
        resampled_base = base_nav[refresh_dates].tolist()
        # 如果调仓日最后一天和净值序列最后一天不一致，则在尾部添加最新净值
        if strategy_nav.index[-1] != refresh_dates[-1]:
            resampled_strategy.append(strategy_nav[-1])
            resampled_base.append(base_nav[-1])
        # 计算调仓超额收益
        resampled_strategy = pd.Series(resampled_strategy)
        resampled_base = pd.Series(resampled_base)
        excess = resampled_strategy.pct_change().dropna() - resampled_base.pct_change().dropna()
        return (excess > 0).sum() / len(excess)
        
    def profit_loss_ratio(self, strategy_nav, base_nav, refresh_dates):
        '''
        计算相对基准盈亏比
        :param strategy_nav: 策略净值
        :param base_nav: 基准净值
        :param refresh_dates: 调仓日期
        :return: 相对基准盈亏比
        '''
        # 抽取调仓日策略和基准的净值
        resampled_strategy = strategy_nav[refresh_dates].tolist()
        resampled_base = base_nav[refresh_dates].tolist()
        # 如果调仓日最后一天和净值序列最后一天不一致，则在尾部添加最新净值
        if strategy_nav.index[-1] != refresh_dates[-1]:
            resampled_strategy.append(strategy_nav[-1])
            resampled_base.append(base_nav[-1])
        # 计算调仓超额收益
        resampled_strategy = pd.Series(resampled_strategy)
        resampled_base = pd.Series(resampled_base)
        excess = resampled_strategy.pct_change().dropna() - resampled_base.pct_change().dropna()
        return - excess[excess > 0].mean() / excess[excess < 0].mean()
    
    def normalize(self,signal:pd.DataFrame)->pd.DataFrame:
        '''
        信号标准化
        : param signal: 信号
        '''
        signal1 = signal.sub(signal.mean(axis=1), axis=0).div(signal.std(axis=1), axis=0)
        # 若标准差为nan, 则保留原数据
        signal1.loc[signal.std(axis=1).isnull(), :] = signal.loc[signal.std(axis=1).isnull(), :]

        return signal1

    def med_std(self,signal:pd.DataFrame, span:int=5)->pd.DataFrame:
        '''
        中位数去极值
        : param signal: 信号
        : span: 取值范围 ub=med+span*mad,lb=med-span*mad
        '''
        signal = signal.copy()
        med = np.nanmedian(signal, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(signal - med), axis=1, keepdims=True)

        ub = med + span * mad
        lb = med - span * mad

        signal1 = signal.values
        for dt in range(signal1.shape[0]):
            if mad[dt] != 0:
                signal1[dt, signal1[dt, :] > ub[dt]] = ub[dt]
                signal1[dt, signal1[dt, :] < lb[dt]] = lb[dt]
        signal.loc[:, :] = signal1

        return signal