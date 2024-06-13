# -*- coding = utf-8 -*-
# @Time : 2024/5/26 13:2
# @Author : ChiXiaoWai
# @File : STL-iTransformer_main.py
# @Project : exp
# 实验场景一
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from neuralforecast import NeuralForecast
from neuralforecast.models import iTransformer
from scipy.signal import savgol_filter
from PyEMD import CEEMDAN
from neuralforecast.losses.pytorch import DistributionLoss, MAE
from iTransformer.models.CustomiTransformer import CustomiTransformer

matplotlib.use('TkAgg')
plt.switch_backend('agg')

# 读取8个数据集并进行预处理
clusters = ['gc19_a', 'gc19_b', 'gc19_c', 'gc19_d', 'gc19_e', 'gc19_f', 'gc19_g', 'gc19_h']
files = ['../data/' + cluster + '.csv' for cluster in clusters]

# 存储划分好的数据集
train_datasets, val_datasets, test_datasets = [], [], []
test_dictionary = {}

# 循环处理每个数据集
for target in ['avgcpu', 'avgmem']:
    for file in files:
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
        # train_datasets.append(df.iloc[:train_len])
        # # CEEMDAN 分解
        # ceemdan = CEEMDAN()
        # imfs = ceemdan(df['y'].values[:train_len])
        # # 重构数据
        # reconstructed_data = np.sum(imfs[:-1], axis=0)  # 不包括最后一个残差项
        #
        # # 替换平滑后的数据
        # df['y'].values[:train_len] = reconstructed_data
        train_set = df.iloc[:train_len]
        # 挨个预测
        # if file == "../data/gc19_a.csv":
        test_set = df.iloc[train_len:]
        # test_datasets.append(test_set)
        file_name = os.path.basename(file)  # 获取文件名部分
        data_name = file_name.split(".")[0]  # 去掉扩展名部分
        test_dictionary[data_name] = test_set

        # 将划分好的数据集存储在列表中
        train_datasets.append(train_set)

    # 合并所有训练集、验证集和测试集
    train_df = pd.concat(train_datasets, ignore_index=True)

    model = iTransformer(h=12,
                         input_size=12,
                         n_series=96,
                         loss=MAE(),
                         val_check_steps=1,
                         hidden_size=64,
                         max_steps=1
                         )

    nf = NeuralForecast(
        models=[model],
        freq='5T'
    )

    # 训练模型
    nf.fit(df=train_df)

    for key in test_dictionary:
        test_df = test_dictionary.get(key)
        # 分块预测
        predictions_list = []
        for i in range(0, len(test_df), 12):
            test_chunk = test_df.iloc[i:i + 12]
            if len(test_chunk) < 12:
                break  # 如果剩余数据不足12个，则停止
            pred_chunk = nf.predict(test_chunk)
            predictions_list.append(pred_chunk)

        # 合并所有预测结果
        Y_hat_df = pd.concat(predictions_list).reset_index(drop=True)
        test_df.reset_index(drop=True, inplace=True)
        Y_hat_df.reset_index(drop=True, inplace=True)
        plot_df = pd.concat([test_df, Y_hat_df], axis=1)
        plot_df = plot_df[plot_df.unique_id == key].drop('unique_id', axis=1)
        df = plot_df[['y', 'iTransformer']].rename(columns={'y': f'{target}-true', 'iTransformer': 'forecast'})
        df.to_csv(f'./experiment1/results/{target}/{key}-{target}-Forecast.csv', index=False)
        plt.figure()
        # 设置绘图风格
        plt.style.use('ggplot')
        plot_df[['y', 'iTransformer']].plot(linewidth=2)
        plt.grid()
        plt.title(f'{key} {target} real vs forecast')
        plt.xlabel('date')
        plt.ylabel(f'{target}')
        plt.legend()
        plt.savefig(f"./experiment1/images/{key}-{target}.png")
