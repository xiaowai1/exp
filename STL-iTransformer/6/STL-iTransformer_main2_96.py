# -*- coding = utf-8 -*-
# @Time : 2024/5/26 13:2
# @Author : ChiXiaoWai
# @File : STL-iTransformer_main.py
# @Project : exp
# 实验场景一
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.tsa.seasonal import STL
from neuralforecast import NeuralForecast
from neuralforecast.models import iTransformer, MLP, LSTM
from scipy.signal import savgol_filter
from neuralforecast.losses.pytorch import MAE, DistributionLoss

matplotlib.use('TkAgg')
plt.switch_backend('agg')

# 读取8个数据集并进行预处理
clusters = ['gc19_a', 'gc19_b', 'gc19_c', 'gc19_d', 'gc19_e', 'gc19_f', 'gc19_g', 'gc19_h']
files = ['../data/' + cluster + '.csv' for cluster in clusters]

# 存储划分好的数据集
train_datasets, val_datasets, test_datasets = [], [], []
seasonal_datasets, trend_datasets, residual_datasets = [], [], []
test_seasonal_dictionary, test_trend_dictionary, test_residual_dictionary = {}, {}, {}

test_index = 7
# 循环处理每个数据集
for target in ['avgcpu', 'avgmem']:
    for i, file in enumerate(files):
        # 读取数据集
        df = pd.read_csv(file, parse_dates=['date'])
        df['date'] = pd.to_datetime(df['date'], unit='us')
        index = 2 if target == 'avgcpu' else 1 if target == 'avgmem' else None
        df.drop(columns=[df.columns[index]], inplace=True)
        # 添加 unique_id 列，用文件名作为每个数据集的唯一标识符
        df['unique_id'] = file.split('/')[-1].split('.')[0]
        df.rename(columns={'date': 'ds'}, inplace=True)
        df.rename(columns={f'{target}': 'y'}, inplace=True)
        # 划分数据集
        train_len = int(len(df) * 0.9)

        # 应用Savitzky-Golay滤波器平滑数据
        df['y'] = savgol_filter(df['y'], window_length=11, polyorder=2)
        # 执行STL分解
        res = STL(df['y'], period=288).fit()
        res.plot()
        trend = res.trend
        seasonal = res.seasonal
        residual = res.resid

        train_seasonal = df.copy()
        train_trend = df.copy()
        train_residual = df.copy()

        # 创建 train_seasonal 数据框并替换前 train_len 行的 'y' 列
        train_seasonal = pd.DataFrame(
            {'ds': train_seasonal['ds'], 'y': seasonal, 'unique_id': train_seasonal['unique_id']})
        train_trend = pd.DataFrame({'ds': train_trend['ds'], 'y': trend, 'unique_id': train_seasonal['unique_id']})
        train_residual = pd.DataFrame(
            {'ds': train_residual['ds'], 'y': residual, 'unique_id': train_seasonal['unique_id']})

        if i == test_index:
            test_seasonal_set = train_seasonal
            test_trend_set = train_trend
            test_residual_set = train_residual
            file_name = os.path.basename(file)  # 获取文件名部分
            data_name = file_name.split(".")[0]  # 去掉扩展名部分
            test_seasonal_dictionary[data_name] = test_seasonal_set
            test_trend_dictionary[data_name] = test_trend_set
            test_residual_dictionary[data_name] = test_residual_set
            continue

        # 将各分量的训练集存储在列表中
        seasonal_datasets.append(train_seasonal)
        trend_datasets.append(train_trend)
        residual_datasets.append(train_residual)

    # 合并所有分量的训练集
    seasonal_df = pd.concat(seasonal_datasets, ignore_index=True)
    trend_df = pd.concat(trend_datasets, ignore_index=True)
    residual_df = pd.concat(residual_datasets, ignore_index=True)

    # 周期项
    model = iTransformer(h=12,
                         input_size=12,
                         n_series=12,
                         loss=MAE(),
                         val_check_steps=1,
                         hidden_size=64,
                         max_steps=1
                         )
    nf = NeuralForecast(
        models=[model],
        freq='5T'
    )

    nf.fit(df=seasonal_df)
    # 残差项
    model2 = iTransformer(h=12,
                          input_size=12,
                          n_series=12,
                          loss=MAE(),
                          val_check_steps=1,
                          hidden_size=64,
                          max_steps=1
                          )
    nf2 = NeuralForecast(
        models=[model2],
        freq='5T'
    )
    nf2.fit(df=residual_df)
    # nf.save(path=f'./experiment1/checkpoints/seasonal/{target}/',
    #         model_index=None,
    #         overwrite=True,
    #         save_dataset=True)
    # 趋势项
    model3 = LSTM(h=12, input_size=96,
                  loss=DistributionLoss(distribution='Normal', level=[80, 90]), max_steps=10)

    nf3 = NeuralForecast(
        models=[model3],
        freq='5T'
    )
    nf3.fit(df=trend_df)

    for key in test_seasonal_dictionary:
        test_seasonal_df = test_seasonal_dictionary.get(key)
        test_trend_df = test_trend_dictionary.get(key)
        test_residual_df = test_residual_dictionary.get(key)
        # 分块预测
        seasonal_predictions_list, trend_predictions_list, residual_predictions_list = [], [], []
        for i in range(0, len(test_seasonal_df), 12):
            test_seasonal_chunk = test_seasonal_df.iloc[i:i + 12]
            test_trend_chunk = test_trend_df.iloc[i:i + 12]
            test_residual_chunk = test_residual_df.iloc[i:i + 12]
            if len(test_seasonal_chunk) < 12:
                break  # 如果剩余数据不足12个，则停止
            pred_chunk_seasonal = nf.predict(test_seasonal_chunk)
            pred_chunk_trend = nf3.predict(test_trend_chunk)
            pred_chunk_residual = nf2.predict(test_residual_chunk)

            seasonal_predictions_list.append(pred_chunk_seasonal)
            trend_predictions_list.append(pred_chunk_trend)
            residual_predictions_list.append(pred_chunk_residual)

        # 合并所有预测结果
        Y_seasonal_df = pd.concat(seasonal_predictions_list).reset_index(drop=True)
        Y_trend_df = pd.concat(trend_predictions_list).reset_index(drop=True)
        Y_residual_df = pd.concat(residual_predictions_list).reset_index(drop=True)

        test_seasonal_df.reset_index(drop=True, inplace=True)
        test_trend_df.reset_index(drop=True, inplace=True)
        test_residual_df.reset_index(drop=True, inplace=True)

        seasonal_plot_df = pd.concat([test_seasonal_df, Y_seasonal_df], axis=1)
        trend_plot_df = pd.concat([test_trend_df, Y_trend_df], axis=1)
        residual_plot_df = pd.concat([test_residual_df, Y_residual_df], axis=1)
        # 周期项
        df = seasonal_plot_df[['y', 'iTransformer']].rename(columns={'y': f'{target}-true', 'iTransformer': 'forecast'})
        df.to_csv(f'./experiment2/results/{target}/seasonal/{key}-{target}-Forecast.csv', index=False)
        plt.figure()
        plt.style.use('ggplot')
        seasonal_plot_df[['y', 'iTransformer']].plot(linewidth=2)
        plt.grid()
        plt.title(f'{key} {target} real vs forecast')
        plt.xlabel('date')
        plt.ylabel(f'{target}')
        plt.legend()
        plt.savefig(f"./experiment2/images/seasonal/{key}-{target}.png")
        # 趋势项
        df = trend_plot_df[['y', 'LSTM']].rename(columns={'y': f'{target}-true', 'LSTM': 'forecast'})
        df.to_csv(f'./experiment2/results/{target}/trend/{key}-{target}-Forecast.csv', index=False)
        plt.figure()
        plt.style.use('ggplot')
        trend_plot_df[['y', 'LSTM']].plot(linewidth=2)
        plt.grid()
        plt.title(f'{key} {target} real vs forecast')
        plt.xlabel('date')
        plt.ylabel(f'{target}')
        plt.legend()
        plt.savefig(f"./experiment2/images/trend/{key}-{target}.png")
        # 残差项
        df = residual_plot_df[['y', 'iTransformer']].rename(columns={'y': f'{target}-true', 'iTransformer': 'forecast'})
        df.to_csv(f'./experiment2/results/{target}/residual/{key}-{target}-Forecast.csv', index=False)
        plt.figure()
        plt.style.use('ggplot')
        residual_plot_df[['y', 'iTransformer']].plot(linewidth=2)
        plt.grid()
        plt.title(f'{key} {target} real vs forecast')
        plt.xlabel('date')
        plt.ylabel(f'{target}')
        plt.legend()
        plt.savefig(f"./experiment2/images/residual/{key}-{target}.png")
