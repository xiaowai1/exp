# -*- coding = utf-8 -*-
# @Time : 2024/5/26 13:24
# @Author : ChiXiaoWai
# @File : iTransformer_main.py
# @Project : exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from neuralforecast import NeuralForecast
from neuralforecast.models import Informer
from neuralforecast.losses.pytorch import DistributionLoss
from models.CustomInformer import CustomInformer
import pywt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pyswarm import pso

matplotlib.use('TkAgg')
plt.switch_backend('agg')

# 读取8个数据集并进行预处理
clusters = ['gc19_a', 'gc19_b', 'gc19_c', 'gc19_d', 'gc19_e', 'gc19_f', 'gc19_g', 'gc19_h']
files = ['../data/' + cluster + '.csv' for cluster in clusters]

# 存储划分好的数据集
train_datasets, val_datasets, test_datasets = [], [], []

# 循环处理每个数据集
for target in ['avgcpu', 'avgmem']:
    for file in files:
        print("start load dataset：" + file + "......")
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
        train_len = int(len(df) * 0.98)
        # 选择要处理的时间序列
        series = df['y'].values[:train_len]

        # 进行 DWT 分解
        wavelet = 'db4'  # 选择 Daubechies 小波
        level = 2
        coeffs = pywt.wavedec(series, wavelet, level=level)
        # 阈值去噪
        threshold = 0.2
        coeffs[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:])

        # 将不同尺度的特征重新组合成特征矩阵
        max_len = max(len(c) for c in coeffs)
        coeffs_padded = [np.pad(c, (0, max_len - len(c)), 'constant') for c in coeffs]
        # 将每个子带信号保存为一个数据框
        features = np.vstack(coeffs_padded).T  # 展平成特征矩阵
        feature_df = pd.DataFrame(features, columns=[f'coeff_{i}' for i in range(features.shape[1])])
        feature_df['unique_id'] = df['unique_id'].values[:len(features)]
        feature_df['ds'] = df['ds'].values[:len(features)]
        feature_df['y'] = df['y'].values[:len(features)]

        train_set = feature_df
        # train_set = df.iloc[:train_len]
        # 挨个预测
        if file == "../data/gc19_a.csv":
            test_set = df.iloc[train_len:]
            test_datasets.append(test_set)

        # 将划分好的数据集存储在列表中
        train_datasets.append(train_set)
        print(f"load dataset {file} success：" + "......")

    # 合并所有训练集、验证集和测试集
    train_df = pd.concat(train_datasets, ignore_index=True)
    test_df = pd.concat(test_datasets, ignore_index=True)

    model = CustomInformer(h=12,
                           input_size=96,
                           hidden_size=16,
                           conv_hidden_size=32,
                           n_head=8,
                           loss=DistributionLoss(distribution='Normal', level=[80, 90]),
                           # futr_exog_list=calendar_cols,
                           scaler_type='robust',
                           learning_rate=1e-3,
                           max_steps=6,
                           val_check_steps=100,
                           )

    nf = NeuralForecast(
        models=[model],
        freq='5T'
    )


    # 目标函数，用于优化
    def objective_function(params, model, train_df):
        model.hidden_size = int(params[0])
        model.conv_hidden_size = int(params[1])
        model.n_head = int(params[2])
        model.learning_rate = params[3]

        # 训练模型
        nf.fit(df=train_df)
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
        plot_df = pd.concat([test_df, Y_hat_df], axis=1)
        plot_df = plot_df[plot_df.unique_id == 'gc19_a'].drop('unique_id', axis=1)
        plot_df.dropna(subset=['CustomInformer'], inplace=True)
        # val_predictions = model.predict(val_df)
        true_values = plot_df['y'].values
        val_predictions = plot_df['CustomInformer'].values

        rmse = np.sqrt(mean_squared_error(true_values, val_predictions))
        mape = mean_absolute_percentage_error(true_values, val_predictions) / 100
        mae = mean_absolute_error(true_values, val_predictions)

        return [rmse, mape, mae]


    def optimize_model(model, train_df):
        lb = [8, 16, 4, 1e-4]  # 参数的下界
        ub = [64, 128, 16, 1e-1]  # 参数的上界

        def objective_function_single(params):
            obj_values = objective_function(params, model, train_df)
            return sum(obj_values)  # 将多个目标值合并为单个值进行优化

        optimal_params, _ = pso(objective_function_single, lb, ub, swarmsize=100, maxiter=200)
        return optimal_params


    optimal_params = optimize_model(model, train_df)

    model.hidden_size = int(optimal_params[0])
    model.conv_hidden_size = int(optimal_params[1])
    model.n_head = int(optimal_params[2])
    model.learning_rate = optimal_params[3]

    # 训练模型
    nf.fit(df=train_df)

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
    plot_df = pd.concat([test_df, Y_hat_df], axis=1)
    plot_df = plot_df[plot_df.unique_id == 'gc19_a'].drop('unique_id', axis=1)
    df = plot_df[['y', 'CustomInformer']].rename(columns={'y': f'{target}-true', 'CustomInformer': 'forecast'})
    df.to_csv('./results/{}-gc19_a-Forecast.csv'.format(target), index=False)
    plt.figure()
    # 设置绘图风格
    plt.style.use('ggplot')
    plot_df[['y', 'CustomInformer']].plot(linewidth=2)
    plt.grid()
    plt.title(f'{target} gc19_a real vs forecast')
    plt.xlabel('date')
    plt.ylabel(f'{target}')
    plt.legend()
    plt.savefig(f"{target}.png")
