# -*- coding = utf-8 -*-
# @Time : 2024/6/4 16:24
# @Author : ChiXiaoWai
# @File : wavelet.py
# @Project : exp
# 读取 CSV 文件
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pywt
import os
plt.switch_backend('agg')
matplotlib.use('TkAgg')


# 读取 CSV 文件
data = pd.read_csv("../data/gc19_a.csv")

# 将数据转换为 numpy 数组
data_array = data.values

# 创建存储图像的目录
for i in range(2):
    target_name = data.columns[i + 1]
    os.makedirs(f"{target_name}_img", exist_ok=True)

    data_to_process = data_array[:, i + 1]

    # 进行 DWT 分解
    wavelet = 'db4'  # 选择 Daubechies 小波
    coeffs = pywt.wavedec(data_to_process, wavelet)

    # 绘制分解后的信号并保存图像
    plt.figure(figsize=(8, 6))
    plt.plot(data_to_process, 'r')
    plt.title("Original signal")
    plt.savefig(f"wavelet_{target_name}_img/original_signal.png")
    plt.close()

    for idx, coeff in enumerate(coeffs):
        plt.figure(figsize=(8, 6))
        plt.plot(coeff, 'g')
        if idx == 0:
            plt.title("Approximation Coefficient")
        else:
            plt.title(f"Detail Coefficient {idx}")
        plt.savefig(f"wavelet_{target_name}_img/coeff_{idx}.png")
        plt.close()