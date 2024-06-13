# -*- coding = utf-8 -*-
# @Time : 2024/6/12 13:25
# @Author : ChiXiaoWai
# @File : CDF.py
# @Project : exp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_absolute_error, mean_squared_error
matplotlib.use('TkAgg')
plt.switch_backend('agg')

experiment = "experiment2"
value = "MSE"
iloc_num_start = 0
iloc_num = 100

informer_df = pd.DataFrame()
LSTM_df = pd.DataFrame()
iTransformer_df = pd.DataFrame()
STL_iTransformer_df = pd.DataFrame()
for file in "h":
    df1 = pd.read_csv(f'../Informer4/{experiment}/results/avgcpu/gc19_{file}-avgcpu-Forecast.csv').iloc[iloc_num_start:iloc_num]
    informer_df = pd.concat([informer_df, df1], axis=0)

    df2 = pd.read_csv(f'../LSTM/{experiment}/results/avgcpu/gc19_{file}-avgcpu-Forecast.csv').iloc[iloc_num_start:iloc_num]
    LSTM_df = pd.concat([LSTM_df, df2], axis=0)

    df3 = pd.read_csv(f'../iTransformer3/{experiment}/results/avgcpu/gc19_{file}-avgcpu-Forecast.csv').iloc[iloc_num_start:iloc_num]
    iTransformer_df = pd.concat([iTransformer_df, df3], axis=0)

    seasonal_df = pd.read_csv(f'../STL-iTransformer/{experiment}/results/avgcpu/seasonal/gc19_{file}-avgcpu-Forecast.csv').iloc[iloc_num_start:iloc_num]
    trend_df = pd.read_csv(f'../STL-iTransformer/{experiment}/results/avgcpu/trend/gc19_{file}-avgcpu-Forecast.csv').iloc[iloc_num_start:iloc_num]
    residual_df = pd.read_csv(f'../STL-iTransformer/{experiment}/results/avgcpu/residual/gc19_{file}-avgcpu-Forecast.csv').iloc[iloc_num_start:iloc_num]
    df4 = seasonal_df.add(trend_df, fill_value=0).add(residual_df, fill_value=0)
    STL_iTransformer_df = pd.concat([STL_iTransformer_df, df4], axis=0)

# 删除 avgcpu-true 列值为 0 的行
informer_df = informer_df[informer_df[f'avgcpu-true'] != 0]
LSTM_df = LSTM_df[LSTM_df[f'avgcpu-true'] != 0]
iTransformer_df = iTransformer_df[iTransformer_df[f'avgcpu-true'] != 0]
STL_iTransformer_df = STL_iTransformer_df[STL_iTransformer_df[f'avgcpu-true'] != 0]

if value == "MSE":
    # 计算MSE
    informer_values = (informer_df[f'avgcpu-true'].values - informer_df['forecast'].values)**2
    LSTM_values = (LSTM_df[f'avgcpu-true'].values - LSTM_df['forecast'].values)**2
    iTransformer_values = (iTransformer_df[f'avgcpu-true'].values - iTransformer_df['forecast'].values)**2
    STL_iTransformer_values = (STL_iTransformer_df[f'avgcpu-true'].values - STL_iTransformer_df['forecast'].values)**2
elif value == "MAE":
    # 计算MAE
    informer_values = np.abs(informer_df[f'avgcpu-true'].values - informer_df['forecast'].values)
    LSTM_values = np.abs(LSTM_df[f'avgcpu-true'].values - LSTM_df['forecast'].values)
    iTransformer_values = np.abs(iTransformer_df[f'avgcpu-true'].values - iTransformer_df['forecast'].values)
    STL_iTransformer_values = np.abs(STL_iTransformer_df[f'avgcpu-true'].values - STL_iTransformer_df['forecast'].values)
else:
    informer_values = np.sqrt((informer_df[f'avgcpu-true'].values - informer_df['forecast'].values) ** 2)
    LSTM_values = np.sqrt((LSTM_df[f'avgcpu-true'].values - LSTM_df['forecast'].values) ** 2)
    iTransformer_values = np.sqrt((iTransformer_df[f'avgcpu-true'].values - iTransformer_df['forecast'].values) ** 2)
    STL_iTransformer_values = np.sqrt((STL_iTransformer_df[f'avgcpu-true'].values - STL_iTransformer_df['forecast'].values) ** 2)

# 对MAE进行排序
informer_mae = np.sort(informer_values)
LSTM_mae = np.sort(LSTM_values)
iTransformer_mae = np.sort(iTransformer_values)
STL_iTransformer_mae = np.sort(STL_iTransformer_values)

# 计算CDF
informer_cdf = np.arange(1, len(informer_mae) + 1) / len(informer_mae)
LSTM_cdf = np.arange(1, len(LSTM_mae) + 1) / len(LSTM_mae)
iTransformer_cdf = np.arange(1, len(iTransformer_mae) + 1) / len(iTransformer_mae)
STL_iTransformer_cdf = np.arange(1, len(STL_iTransformer_mae) + 1) / len(STL_iTransformer_mae)

# 绘制CDF图
plt.figure(figsize=(10, 6))
# 绘制折线图
plt.plot(LSTM_mae, LSTM_cdf, linestyle='-', color='orange', label='LSTM', marker='P', linewidth=1)
plt.plot(informer_mae, informer_cdf, linestyle='-', color='royalblue', label='informer', marker='o', linewidth=1)
plt.plot(iTransformer_mae, iTransformer_cdf, linestyle='-', color='green', label='iTransformer', marker='*', linewidth=1)
plt.plot(STL_iTransformer_mae, STL_iTransformer_cdf, linestyle='-', color='hotpink', label='STL_iTransformer', marker='p', linewidth=1)
plt.legend(loc='lower right')
plt.xlim(0, max(np.max(informer_mae), np.max(LSTM_mae), np.max(iTransformer_mae),
                np.max(STL_iTransformer_mae)) * 1.1)
# plt.xlim(0, 0.125)
# if target == "avgmem":
#     plt.xlim(0, 0.025)
plt.xlabel(f'{value}')
plt.ylabel('CDF')
plt.title(f'CDF of {value} between True and Predicted Values')
plt.savefig(f"./{experiment}-{value}-avgcpu-cdf.png")




informer_df = pd.DataFrame()
LSTM_df = pd.DataFrame()
iTransformer_df = pd.DataFrame()
STL_iTransformer_df = pd.DataFrame()
for file in "bc":
    df1 = pd.read_csv(f'../Informer4/{experiment}/results/avgmem/gc19_{file}-avgmem-Forecast.csv').iloc[iloc_num_start:iloc_num]
    informer_df = pd.concat([informer_df, df1], axis=0)

    df2 = pd.read_csv(f'../LSTM/{experiment}/results/avgmem/gc19_{file}-avgmem-Forecast.csv').iloc[iloc_num_start:iloc_num]
    LSTM_df = pd.concat([LSTM_df, df2], axis=0)

    df3 = pd.read_csv(f'../iTransformer3/{experiment}/results/avgmem/gc19_{file}-avgmem-Forecast.csv').iloc[iloc_num_start:iloc_num]
    iTransformer_df = pd.concat([iTransformer_df, df3], axis=0)

    seasonal_df = pd.read_csv(f'../STL-iTransformer/{experiment}/results/avgmem/seasonal/gc19_{file}-avgmem-Forecast.csv').iloc[iloc_num_start:iloc_num]
    trend_df = pd.read_csv(f'../STL-iTransformer/{experiment}/results/avgmem/trend/gc19_{file}-avgmem-Forecast.csv').iloc[iloc_num_start:iloc_num]
    residual_df = pd.read_csv(f'../STL-iTransformer/{experiment}/results/avgmem/residual/gc19_{file}-avgmem-Forecast.csv').iloc[iloc_num_start:iloc_num]
    df4 = seasonal_df.add(trend_df, fill_value=0).add(residual_df, fill_value=0)
    STL_iTransformer_df = pd.concat([STL_iTransformer_df, df4], axis=0)

# 删除 avgcpu-true 列值为 0 的行
informer_df = informer_df[informer_df[f'avgmem-true'] != 0]
LSTM_df = LSTM_df[LSTM_df[f'avgmem-true'] != 0]
iTransformer_df = iTransformer_df[iTransformer_df[f'avgmem-true'] != 0]
STL_iTransformer_df = STL_iTransformer_df[STL_iTransformer_df[f'avgmem-true'] != 0]

if value == "MSE":
    # 计算MSE
    informer_values = (informer_df[f'avgmem-true'].values - informer_df['forecast'].values)**2
    LSTM_values = (LSTM_df[f'avgmem-true'].values - LSTM_df['forecast'].values)**2
    iTransformer_values = (iTransformer_df[f'avgmem-true'].values - iTransformer_df['forecast'].values)**2
    STL_iTransformer_values = (STL_iTransformer_df[f'avgmem-true'].values - STL_iTransformer_df['forecast'].values)**2
elif value == "MAE":
    # 计算MAE
    informer_values = np.abs(informer_df[f'avgmem-true'].values - informer_df['forecast'].values)
    LSTM_values = np.abs(LSTM_df[f'avgmem-true'].values - LSTM_df['forecast'].values)
    iTransformer_values = np.abs(iTransformer_df[f'avgmem-true'].values - iTransformer_df['forecast'].values)
    STL_iTransformer_values = np.abs(STL_iTransformer_df[f'avgmem-true'].values - STL_iTransformer_df['forecast'].values)
else:
    informer_values = np.sqrt((informer_df[f'avgmem-true'].values - informer_df['forecast'].values) ** 2)
    LSTM_values = np.sqrt((LSTM_df[f'avgmem-true'].values - LSTM_df['forecast'].values) ** 2)
    iTransformer_values = np.sqrt((iTransformer_df[f'avgmem-true'].values - iTransformer_df['forecast'].values) ** 2)
    STL_iTransformer_values = np.sqrt((STL_iTransformer_df[f'avgmem-true'].values - STL_iTransformer_df['forecast'].values) ** 2)

# 对MAE进行排序
informer_mae = np.sort(informer_values)
LSTM_mae = np.sort(LSTM_values)
iTransformer_mae = np.sort(iTransformer_values)
STL_iTransformer_mae = np.sort(STL_iTransformer_values)

# 计算CDF
informer_cdf = np.arange(1, len(informer_mae) + 1) / len(informer_mae)
LSTM_cdf = np.arange(1, len(LSTM_mae) + 1) / len(LSTM_mae)
iTransformer_cdf = np.arange(1, len(iTransformer_mae) + 1) / len(iTransformer_mae)
STL_iTransformer_cdf = np.arange(1, len(STL_iTransformer_mae) + 1) / len(STL_iTransformer_mae)

# 绘制CDF图
plt.figure(figsize=(10, 6))

# 绘制折线图
plt.plot(LSTM_mae, LSTM_cdf, linestyle='-', color='orange', label='LSTM', marker='P', linewidth=1)
plt.plot(informer_mae, informer_cdf, linestyle='-', color='royalblue', label='informer', marker='o', linewidth=1)
plt.plot(iTransformer_mae, iTransformer_cdf, linestyle='-', color='green', label='iTransformer', marker='*', linewidth=1)
plt.plot(STL_iTransformer_mae, STL_iTransformer_cdf, linestyle='-', color='hotpink', label='STL_iTransformer', marker='p', linewidth=1)
plt.legend(loc='lower right')
plt.xlim(0, max(np.max(informer_mae), np.max(LSTM_mae), np.max(iTransformer_mae),
                np.max(STL_iTransformer_mae)) * 1.1)

# plt.xlim(0, 0.125)
# if target == "avgmem":
#     plt.xlim(0, 0.025)
plt.xlabel(f'{value}')
plt.ylabel('CDF')
plt.title(f'CDF of {value} between True and Predicted Values')

plt.savefig(f"./{experiment}-{value}-avgmem-cdf.png")
