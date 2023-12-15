'''
@ Author: lxxu
@ Date: 2023-12-08 19:39:00
@ Description: 可视化
@ Email: lixinxu@stu.pku.edu.cn
'''

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties

import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")

class Presenter():
    def __init__(self, save:bool=False, path:str='./'):
        '''
        初始化
        : param save: 是否保存图像,默认为False
        : param path: 图像保存路径,默认与用户代码在相同路径
        '''
        self.save = save
        self.path = path
        self.result_file = {}


    def load_results(self, result_file:dict={}):
        '''
        初始化
        : param result_file: 回测结果文件, 默认格式为:
            dic({'signal':pd.DataFrame,'stats_results':pd.DataFrame,'stats_series':pd.DataFrame,
            'clsfy_nav':pd.DataFrame,'clsfy_perf':pd.DataFrame,'ls_nav':pd.DataFrame,'ls_perf':pd.DataFrame})
        '''
        for k in result_file.keys():
            if k not in ['signal','stats_results','stats_series','clsfy_nav','clsfy_perf',
                         'ls_nav','ls_perf'] or type(result_file[k])!=pd.DataFrame:
                raise ValueError('Wrong format of result_file!')
        self.result_file = result_file
        

    def multi_plot(self, result:dict={},title:str=''):
        '''
        绘制多图
        : param result: 回测结果文件, 默认格式为:
            dic({'signal':pd.DataFrame,'stats_results':pd.DataFrame,'stats_series':pd.DataFrame,
            'clsfy_nav':pd.DataFrame,'clsfy_perf':pd.DataFrame,'ls_nav':pd.DataFrame,'ls_perf':pd.DataFrame})'
        : param save: 是否保存图像,默认为False
        : param path: 图像保存路径,默认与用户代码在相同路径
        '''
        if len(result)==0:
            if len(self.result_file) != 0:
                result = self.result_file
            else:
                raise ValueError('Please load result_file first!')
            
        # 设置图像风格
        plt.style.use('seaborn-whitegrid')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.edgecolor'] = 'lightgrey'
        plt.rcParams['grid.color'] = 'lightgrey'
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.5
        # 设置标题字体属性
        title_formatdic={'fontsize': 17, 'fontweight': 'bold', 'family':'KaiTi'}

        # 创建一个3x3的子图网格，使用gridspec
        fig = plt.figure(figsize=(20, 20))
        gs = gridspec.GridSpec(3, 3)
        # 自定义大标题的字体属性
        title_font = FontProperties(weight='bold', size=25, family='SimHei')
        fig.suptitle(f"{title}回测结果\n\n", fontproperties=title_font)
        
        plt.rcParams['font.sans-serif'] = ['KaiTi']  # 设置中文字体为楷体

        # 创建子图
        ax1 = plt.subplot(gs[0, 0]);ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[0, 2]);ax4 = plt.subplot(gs[1, 0])
        ax5 = plt.subplot(gs[1, 1:]);ax6 = plt.subplot(gs[2, 0])
        ax7 = plt.subplot(gs[2, 1:])

        # 设置颜色
        colors=['#ff6b6b', '#a3cf62', '#6a89cc', '#82ccdd', '#ffb142', '#a5b1c2']

        # 图1：因子分布箱图
        merged_factor_data = pd.Series(result['signal'].values.flatten())
        sns.boxplot(merged_factor_data, ax=ax1, color=colors[0], boxprops={'edgecolor': 'white'})
        # sns.kdeplot(merged_factor_data, shade=True, ax=ax1, color=colors[0])
        ax1.set_title('信号分布箱图', fontdict=title_formatdic)

        # 图2：IC图
        col_list = ['IC均值','IC标准差','rank_IC均值','rank_IC标准差','IC_IR']
        result['stats_results'][col_list].plot(kind='bar',ax=ax2,color=colors)
        ax2.set_title('IC & rank_IC & IC_IR', fontdict=title_formatdic)
        # 在柱上标注数值
        for i,p in enumerate(ax2.patches):
            if result['stats_results'][col_list[i]].iloc[0]>=0:
                ax2.annotate(f'{p.get_height():.3f}\n', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=12, color=colors[i], xytext=(0, 5),
                            textcoords='offset points')
            else:
                ax2.annotate(f'{p.get_height():.3f}\n', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=12, color=colors[i], xytext=(0, -15),
                            textcoords='offset points')

        # 图3：累计IC rank_IC时间序列
        result['stats_series'][['累计IC','累计rank_IC']].plot(ax=ax3,color=colors)
        ax3.fill_between(result['stats_series'].index, 0, result['stats_series']['累计rank_IC'], color=colors[1], alpha=0.3)
        ax3.fill_between(result['stats_series'].index, 0, result['stats_series']['累计IC'], color=colors[0], alpha=0.3)
        ax3.set_title('累计IC & 累计rank_IC', fontdict=title_formatdic)

        # 图4：分层回测净值
        result['clsfy_nav'].loc[:,'分层1相对净值':].plot(ax=ax4,color=colors)
        ax4.set_title('[分层回测] 相对(基准)净值', fontdict=title_formatdic)
        
        # 图5：分层回测结果
        result['clsfy_perf'].T.plot(kind='bar',ax=ax5, color=colors)
        ax5.set_title('[分层回测] 表现', fontdict=title_formatdic)
        ax5.set_xticklabels(ax5.get_xticklabels(), rotation=30, ha='right')

        # 图6：多空回测净值
        result['ls_nav'].loc[:,'多头相对净值':].plot(ax=ax6,color=colors)
        ax6.set_title('[多空回测] 相对(基准)净值', fontdict=title_formatdic)
        
        # 图7：多空回测结果
        result['ls_perf'].T.plot(kind='bar',ax=ax7,color=colors)
        ax7.set_title('[多空回测] 表现', fontdict=title_formatdic)
        ax7.set_xticklabels(ax5.get_xticklabels(), rotation=30, ha='right')

        plt.tight_layout()
        if self.save:
            plt.savefig(self.path+f'{title}回测结果.png')
        plt.show()
            

    def signal_box(self, signal:pd.DataFrame=pd.DataFrame(),title:str=''):
        '''
        绘制信号分布箱图
        : param signal: 信号数据, 默认为None, 会从result_file中读取
        : param save: 是否保存图像,默认为False
        : param path: 图像保存路径,默认与用户代码在相同路径
        '''
        if len(signal)==0:
            if len(self.result_file)!=0:
                signal = self.result_file['signal']
            else:
                raise ValueError('Please load result_file or signal first!')
        # 绘制信号分布图
        fig,axes = plt.subplots(1,1, figsize=(12, 6))
        merged_factor_data = pd.Series(signal.values.flatten())
        sns.boxplot(merged_factor_data, color='#ff6b6b', boxprops={'edgecolor': 'white'})
        # sns.kdeplot(merged_factor_data, shade=True, ax=ax1, color=colors[0])
        plt.title('信号分布箱图', fontdict={'fontsize': 17, 'fontweight': 'bold', 'family':'KaiTi'})
        # 保存图像
        if self.save:
            plt.savefig(self.path+f'{title}信号分布箱图.png')
        plt.show()
        


    def plot_heatmap(self, data_df:pd.DataFrame,title:str=''):
        '''
        绘制热力图
        : param data_df: 需要绘制热力图的DataFrame
        : param title: 图像标题 默认为'相关性热力图'
        '''
        plt.figure(figsize=(12, 10))
        sns.heatmap(data_df.astype(float), cmap='RdBu_r', linewidths=0.5, annot=True, fmt='.2f')
        plt.title('相关性热力图', fontdict={'fontsize': 17, 'fontweight': 'bold', 'family':'KaiTi'})
        if self.save:
            plt.savefig(self.path+f'{title}相关性热力图.png') 
        plt.show()   

