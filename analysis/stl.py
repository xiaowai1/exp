# -*- coding = utf-8 -*-
# @Time : 2024/6/10 21:48
# @Author : ChiXiaoWai
# @File : stl.py
# @Project : exp
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.tsa.seasonal import STL
matplotlib.use('TkAgg')
plt.switch_backend('agg')

df = pd.read_csv("../data/gc19_a.csv", parse_dates=['date'])
df['date'] = pd.to_datetime(df['date'], unit='us')
df.set_index('date', inplace=True)

res = STL(df['avgcpu'], period=288 * 7).fit()

# 绘制原始数据图
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['avgcpu'], label='Original Data')
plt.title('Original Data')
plt.xlabel('Date')
plt.ylabel('avgcpu')
plt.legend()
plt.savefig("./stl_images/origin.png")

# 绘制周期分量图
plt.figure(figsize=(10, 6))
plt.plot(df.index, res.seasonal, label='')
plt.title('Seasonal Component')
plt.xlabel('Date')
plt.ylabel('avgcpu')
plt.legend()
plt.savefig("./stl_images/Seasonal.png")

# 绘制趋势分量图
plt.figure(figsize=(10, 6))
plt.plot(df.index, res.trend, label='Trend')
plt.title('Trend Component')
plt.xlabel('Date')
plt.ylabel('avgcpu')
plt.legend()
plt.savefig("./stl_images/Trend.png")

# 绘制残差分量图
plt.figure(figsize=(10, 6))
plt.plot(df.index, res.resid, label='Residual')
plt.title('Residual Component')
plt.xlabel('Date')
plt.ylabel('avgcpu')
plt.legend()
plt.savefig("./stl_images/Residual.png")
