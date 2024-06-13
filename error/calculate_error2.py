# -*- coding = utf-8 -*-
# @Time : 2024/5/20 14:33
# @Author : ChiXiaoWai
# @File : calculate_error.py
# @Project : DLinear
# @Description : 误差计算
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.switch_backend('agg')


for target in ['avgcpu', 'avgmem']:
    total_mse = 0
    total_mae = 0
    # total_mape = 0
    total_rmse = 0
    for file in "abcdefgh":
        y_true, y_pred = 0, 0
        for stl in ['seasonal', 'trend', 'residual']:

            # 从 CSV 文件中加载数据
            # df = pd.read_csv('../iTransformer/results/avgmem-gc19_a-Forecast.csv')  # 替换为你的 CSV 文件路径
            df = pd.read_csv(f'../STL-iTransformer/experiment1/results/{target}/{stl}/gc19_{file}-{target}-Forecast.csv')

            # 提取实际值和预测值列，并转换为 NumPy 数组
            # y_true = df['avgmem-true'].values
            y_true = y_true + df[f'{target}-true'].values
            y_pred = y_pred + df['forecast'].values


        # 删除含有 NaN 或 Inf 值的行
        valid_idx = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_true) & ~np.isinf(y_pred)
        y_true = y_true[valid_idx]
        y_pred = y_pred[valid_idx]

        # 计算均方误差 (MSE)
        mse = np.mean((y_true - y_pred)**2)
        total_mse += mse
        # 计算平均绝对误差 (MAE)
        mae = mean_absolute_error(y_true, y_pred)
        total_mae += mae
        # # 计算平均绝对百分比误差 (MSE)
        # mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        # total_mape += mape
        # 计算均方根误差 (RMSE)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        total_rmse += rmse
        # 计算对称平均绝对百分比误差 (SMAPE)
        # smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

        plt.figure()
        plt.style.use('ggplot')
        # 生成 x 轴数据（假设 x 轴是样本编号）
        x = np.arange(1, len(y_pred) + 1)

        # 绘制折线图
        plt.plot(x, y_pred, label='Predicted')
        plt.plot(x, y_true, label='True')
        plt.title(f'{file}-{target} real vs forecast')
        plt.xlabel('date')
        plt.ylabel(f'{target}')
        plt.legend()
        plt.savefig(f"../STL-iTransformer/experiment1/{file}-{target}.png")

    # 输出结果
    # print("Mean Squared Error (MSE):", mse)
    print(f"{target} Error: ")
    print("Mean Squared Error (MSE):", total_mse / 8)
    print("Mean Absolute Error (MAE):", total_mae / 8)
    # print("Mean Absolute Percentage Error (MSE):", total_mape / 8)
    print("Root Mean Squared Error (RMSE):", total_rmse / 8)
    # print("Symmetric Mean Absolute Percentage Error (SMAPE):", smape)
