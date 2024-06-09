# -*- coding = utf-8 -*-
# @Time : 2024/6/7 9:54
# @Author : ChiXiaoWai
# @File : analysis.py
# @Project : exp
import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
plt.switch_backend('agg')

# 读取数据
df = pd.read_csv('../data/gc19_a.csv')  # 替换为您的数据文件路径

# 提取avgcpu列数据进行分析操作
column_data = df['avgcpu']
column_data2 = df['avgmem']


# 相关性分析
def analyze_correlation(data1, data2):
    """
    分析两列数据的相关性
    参数:
    - data1: 第一列数据，可以是一个一维数组或列表
    - data2: 第二列数据，可以是一个一维数组或列表
    """
    # 将数据转换为NumPy数组
    data1 = np.array(data1)
    data2 = np.array(data2)

    # 计算Pearson相关系数
    pearson_corr, _ = stats.pearsonr(data1, data2)

    # 计算Spearman相关系数
    spearman_corr, _ = stats.spearmanr(data1, data2)

    # 打印相关系数
    print("Pearson相关系数: ", pearson_corr)
    print("Spearman相关系数: ", spearman_corr)


# 趋势性分析
def calculate_trend(data1, data2):
    # 创建 DataFrame

    # 提取自变量和因变量
    x = np.array(data1).reshape(-1, 1)
    y = np.array(data2).reshape(-1, 1)

    # 使用线性回归模型拟合数据
    model = LinearRegression()
    model.fit(x, y)

    # 提取斜率和截距
    slope = model.coef_[0]
    intercept = model.intercept_

    # 返回趋势性分析结果
    result = {
        'slope': slope,
        'intercept': intercept,
    }
    print(result)

# 傅里叶变换
def FourierDataPlot(column_data):
    # 计算傅里叶变换及频谱
    fft = np.fft.fft(column_data)
    freq = np.fft.fftfreq(len(column_data))
    plt.plot(freq, np.abs(fft))
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Frequency Spectrum')
    plt.grid(True)
    plt.savefig('./analysis/fourier_plot.png')


# ACF和PACF
def acfDataPlot(data):
    # 设置 seaborn 风格
    sns.set(style='whitegrid')

    # 创建 ACF 图
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plot_acf(data, ax=ax1, lags=40)
    ax1.set_title('自相关函数 (ACF)')
    ax1.set_xlabel('滞后')
    ax1.set_ylabel('自相关')

    # 计算置信区间
    n = len(data)
    conf_95 = 1.96 / (n ** 0.5)  # 95% 置信区间
    conf_99 = 2.576 / (n ** 0.5)  # 99% 置信区间
    ax1.axhline(y=conf_95, linestyle='--', color='r', label='95% 置信区间')
    ax1.axhline(y=-conf_95, linestyle='--', color='r')
    ax1.axhline(y=conf_99, linestyle='--', color='g', label='99% 置信区间')
    ax1.axhline(y=-conf_99, linestyle='--', color='g')
    ax1.legend()

    # 保存 ACF 图像文件
    plt.savefig('./analysis/acf_plot.png')
    plt.close()

    # 创建 PACF 图
    fig, ax2 = plt.subplots(figsize=(10, 6))
    plot_pacf(data, ax=ax2, lags=40)
    ax2.set_title('偏自相关函数 (PACF)')
    ax2.set_xlabel('滞后')
    ax2.set_ylabel('偏自相关')

    # 计算置信区间
    ax2.axhline(y=conf_95, linestyle='--', color='r', label='95% 置信区间')
    ax2.axhline(y=-conf_95, linestyle='--', color='r')
    ax2.axhline(y=conf_99, linestyle='--', color='g', label='99% 置信区间')
    ax2.axhline(y=-conf_99, linestyle='--', color='g')
    ax2.legend()

    # 保存 PACF 图像文件
    plt.savefig('./analysis/pacf_plot.png')
    plt.close()


calculate_trend(column_data, column_data2)
# analyze_correlation(column_data, column_data2)
# acfDataPlot(column_data)
# FourierDataPlot(column_data)