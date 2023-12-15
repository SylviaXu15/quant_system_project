# Readme

## I. Structure

* my_repo/
  * newdata/
    * `tot_mv.feather`, `index_basic.feather`, `index_close.feather`, `stock_basic.feather`
  * mysytem/
    * `DataLoader.py` 读取数据+数据处理
    * `SignalGenerator.py` 生成信号+信号处理
    * `Backtest.py` 回测
    * `Presentation.py` 可视化 
  * result_imgs/
  * `test.ipynb`
  * `Readme.md`
  * `newdata_dowloader.ipynb`

其中mysystem下是系统的四个模块，test.ipynb是用户端文件，newdate是新找的数据，newdata_downloader是下载新数据的代码，result_imgs文件夹下存放回测结果的图片

## II. Data

### 2.1 Data

助教提供的数据，一个量价数据文件，三张财务报表数据文件，以及一些辅助文件

### 2.2 NewData

#### 2.2.1 数据来源

下载新数据代码见本md同级目录下的`newdata_download.ipynb`

数据来源：tushare （token是我买的，助教review的时候可能已经过期了x）

#### 2.2.2 数据组成

`stock_basic.feather`

全A股股票基本信息，包含ts_code, name, industry, list_date字段，其中industry行业分类用于行业中性化。

`index_basic.feather`

指数基本信息，包含8000种指数的ts_code, name, market, publisher, category, base_date, base_point, list_date字段。

`index_close.feather`

指数收盘价数据，由于index_basic种的8000太多，下载耗时太长，所以这里我只下载保存了310种SSE（上交所）的指数品种在2020-2022年这3年间的日频收盘价数据。

`tot_mv.feather`

全A股股票2020-2023年间的总市值数据，用于市值中性化。

## III. MySystem

<u>以下是简单介绍，具体模块、类和接口设置以及函数的参数、返回值信息请使用`help`方法</u>。示例如下：

```python
from mysystem import DataLoader
help(DataLoader)   # 查看模块介绍
help(DataLoader.DataLoader)   # 查看类介绍
help(DataLoader.DataLoader.datafile_list)   # 查看类方法介绍
```

### 3.1 DataLoader

包含一个类，`DataLoader`

先导入库和实例化类

```python
from mysystem import DataLoader
loader = DataLoader.DataLoader()
```

然后可以使用类方法

```python
# 例：读取复权收盘价
closeadj = loader.load_pricevol_data(genre='close',adjusted=True)
```

* 类属性

  > path: 文件保存路径
  >
  > startdate/ enddate: 读取数据的时间范围
  >
  > pricevolume: data文件夹中的量价数据表
  >
  > balance/ cashflow/ income: data文件夹中的三张财务报表
  >
  > stockpool: 量价与基本面数据股票池的交集
  >
  > tradedate: 选定时间范围内的交易日列表

* 类方法

  > **【datafile_list】**
  >
  > * 返回data文件夹中的文件名目录
  > * 无参数，返回文件名列表
  >
  > **【get_fundamental_list】**
  >
  > * 查看基本面数据列表
  > * 无参数，返回三张财务报表名为键，表中包含数据组成的列表为值的字典
  >
  > **【load_pricevol_data】**
  >
  > * 读取量价数据 （高开低收价格以及交易量/额）
  > * 可以通过adjusted设置是否复权（仅对价格数据有效）
  > * 返回index为时间戳，columns为股票的数据列表
  >
  > **【load_fundamental_data】**
  >
  > * 读取基本面数据
  > * 通过genre指定读取的数据名称，通过fill控制是否向前填充缺失值
  >
  > **【load_newdata】**
  >
  > * 读取newdata文件夹中的数据
  > * 通过genre指定读取的文件名，返回读取的数据pd.DataFrame

### 3.2 SignalGenerator

包含两个类`SignalGenerator`和`SignalProcessor`

#### 3.2.1 SignalGenerator

主要用于自动生成信号，内置了动量策略（如果有其它需要程序员可自行补充）

* 类属性

  > startdate/ enddate: 生成信号的时间范围
  >
  > pricevolume: data文件夹中的量价数据表
  >
  > balance/ cashflow/ income: data文件夹中的三张财务报表
  >
  > stockpool: 量价与基本面数据股票池的交集
  >
  > tradedate: 选定时间范围内的交易日列表

* 类方法

  > **【momentum】**
  >
  > * 自动生成动量策略的信号
  > * interval为str，格式为'5D', '1M'，用于设置生成多长间隔的动量策略信号，is_tradingdate设置是否按交易日计算（否则按自然日）

#### 3.2.2 SignalProcessor

提供信号处理方法：如中性化，去极值

* 类方法

  > **【MV_Neutral】**
  >
  > * 市值中性化，参数为signal (pd.DataFrame)，返回市值中性化后的signal
  >
  > **【Ind_Neutral】**
  >
  > * 行业中性化，参数为signal (pd.DataFrame)，返回行业中性化后的signal
  >
  > **【MAD_Std】**
  >
  > * 中位数去极值
  > * 参数为signal，表示原始信号。limit是一个默认值为5的整数，用于确定异常值的范围
  > * 返回中位数去极值后的signal
  >
  > **【Rank_Std】**
  >
  > * Rank标准化，参数为signal，表示原始信号，返回rank后的signal
  >
  > **【ZScore_Std】**
  >
  > * Z-score标准化，对每日的信号减去当日均值后除以当日标准差，返回处理后的signal

#### 3.3.3 示例

``` python
from mysystem import SignalGenerator
# 实例化信号生成接口
gen = SignalGenerator.SignalGenerator()
mmt_5d = gen.momentum(interval='5D', is_tradingdate=True)

processor = SignalGenerator.SignalProcessor()
# 市值中性化：以5日动量因子为例
mmt_5d_mc = processor.MV_Neutral(mmt_5d)
```

### 3.3 Backtest

用于信号回测，主要包括两个类：`SignalBacktest`和`BacktestTool`

其中策略回测应当使用`SignalBacktest`，而`BacktestTool`主要是为`SignalBacktest`提供工具支持，用户也可根据需求自行调用

#### 3.3.1 SignalBacktest

* 类属性

  > startdate/ enddate: 回测时间范围
  >
  > stock_close: 股票收盘价
  >
  > stock_close_adj: 股票复权收盘价
  >
  > tradedates: 回测时间范围内的交易日列表
  >
  > index_close_agg: 310种上交所指数产品收盘价
  >
  > index_close: 上证指数收盘价
  >
  > testtools: BacktestTool类的实例化对象
  >
  > presenter: Presentation库的Presentation类的实例化对象

* 类方法 - 单策略

  > **【load_signal】**
  >
  > * 将回测信号读入
  > * 参数：signal_type指定读入信号的方式，有'df'或者'path'。参数signal_file仅当signal_type为'path'的时候读取对应路径的文件。参数signal_df仅当signal_type为'df'的时候读入signal (pd.DataFrame)
  >
  > **【change_base】**
  >
  > * 修改回测的市场基准指数，默认用上证指数作为市场基准
  > * 参数base是修改成的指数名，可以修改成index_close中的310种指数
  >
  > **【strategy_eval】**
  >
  > * 策略回测，无参数
  > * 在函数内调用`stats_test`,`clsfy_backtest`,`ls_backtest`函数，返回数据表名字为键，数据为值的字典
  >
  > **【stats_test】**
  >
  > * 计算统计性指标，返回两个DataFrame类型的表，包括
  >   * stats_result: IC/rank_IC均值/标准差,IC_IR,|t|均值,|t|>2占比,t均值,因子收益率均值,R方均值
  >   * stats_series: (累计)IC, (累计)rank_IC, t, 因子收益率, R方时间序列
  > * 参数signal可以输入一个外来的pd.DataFrame进行计算，如果没有的话，默认使用load_signal中load的signal，如果两者都没有则报错。参数freq指定重采样的频率。
  >
  > **【clsfy_test】**
  >
  > * 分层回测，返回三个DataFrame类型的表，包括
  >   * nav: 各分层净值, 基准净值, 各分层相对净值序列
  >   * port: 各分层持仓权重序列 list[pd.DataFrame]
  >   * performance: 分层1~5年化收益率,年化波动率,夏普比率,最大回撤,年化超额收益率,超额收益年化波动率, 信息比率,超额收益最大回撤,调仓胜率,相对基准盈亏比
  > * 参数signal可以输入一个外来的pd.DataFrame进行计算，如果没有的话，默认使用load_signal中load的signal，如果两者都没有则报错。参数freq指定重采样的频率。还可以指定时间范围startdate, enddate，以及参数layer_number设置分层层数，fee控制手续费。
  >
  > **【ls_backtest】**
  >
  > * 多空回测，返回三个DataFrame类型的表，包括
  >   * nav: 多头, 空头, 多空, 基准, 多头相对净值, 空头相对净值, 多空相对净值序列
  >   * port: 多头/空头持仓权重序列 list[pd.DataFrame]
  >   * performance: 多头/空头/多空年化收益率,年化波动率,夏普比率,最大回撤,年化超额收益率,超额收益年化波动率, 信息比率,超额收益最大回撤,调仓胜率,相对基准盈亏比
  > * 参数signal可以输入一个外来的pd.DataFrame进行计算，如果没有的话，默认使用load_signal中load的signal，如果两者都没有则报错。参数freq指定重采样的频率。还可以指定时间范围startdate, enddate，以及参数layer_number设置分层层数，fee控制手续费。
  >
  >   **【nav_perf】**
  >
  > * 根据净值序列计算业绩表现指标

* 类方法 - 多策略

  > **【corr_matrix】**
  >
  > * 参数signal_dict为多种策略的信号字典，而plot, save, path控制是否绘图以及保存，如果绘图则显示Presentation自定义库中类方法绘制的相关性热力图
  > * 返回相关性矩阵的pd.DataFrame
  >
  > **【strategy_merge】**
  >
  > * 按照参数weight_list的权重加权求和标准化后的signal_dict中的信号值，得到的结果进行标准化后返回

* 类方法-其他

  > **【save_results】**
  >
  > * 保存strategy_eval回测结果到指定路径，save_as参数控制保存文件类型，可以选择feather, pickle, excel，默认为feather
  >
  > **【plot_results】**
  >
  > * 调用Presentation模块中Presentater类的实例对象，可视化回测结果

#### 3.3.2 BacktestTool

主要是对StrategyBacktest提供支持，包括cal_nav (净值计算), excess_statis (计算超额收益业绩指标), normal_statis (计算基本业绩指标，包括年化收益，年化波动，夏普比率，最大回撤)，annualized return (计算年化收益), annualized volatility (计算年化波动), sharpe ratio (计算夏普比例), max_drawdown (计算最大回撤), win_rate (计算调仓胜率), profit_loss_ratio (计算相对基准盈亏比), normalize (标准化), med_std (中位数去极值)

### 3.4 Presentation

包含一个类`Presenter`，用于实现可视化。

#### 3.4.1 Presenter

* 类属性

  > save: 可视化结果是否保存
  >
  > path: 可视化结果保存路径
  >
  > result_file: 用于可视化的回测结果文件，默认为{}

* 类方法

  > **【load_results】**
  >
  > * 参数为回测结果字典，设置self.result_file为参数中的结果字典
  >
  > **【multi_plot】**
  >
  > * 参数result_file，可以选择使用self.result_file，也可以利用这个参数对其他回测结果进行可视化
  > * 参数title，默认为''，可以输入策略名称字符串xxx，最后回测结果图表标题为“xxx回测结果”
  >
  > **【signal_box】**
  >
  > * 可视化信号分布。参数signal允许通过参数可视化其他的策略信号，默认情况下展示self.result_file中的signal的分布
  > * 参数title，默认为''，可以输入策略名称字符串xxx，最后图表标题为“xxx信号分布”
  >
  > **【plot_heatmap】**
  >
  > * 参数为一个pd.DataFrame(比如说相关性矩阵)，绘制对应的热力图
  > * 参数title，默认为''，可以输入名称字符串xxx，最后图表标题为“xxx相关性热力图”


