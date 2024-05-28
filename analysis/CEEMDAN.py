# -*- coding = utf-8 -*-
# @Time : 2024/5/28 16:33
# @Author : ChiXiaoWai
# @File : CEEMDAN.py
# @Project : exp
import pandas as pd
from PyEMD import CEEMDAN
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.switch_backend('agg')
matplotlib.use('TkAgg')

# 读取 CSV 文件
data = pd.read_csv("../data/gc19_a.csv")

# 将数据转换为 numpy 数组
data_array = data.values

for i in range(2):
    data_to_process = data_array[:, i + 1]

    target_name = data.columns[i + 1]

    # 进行 CEEMDAN 分解
    ceemdan = CEEMDAN()
    IMF = ceemdan(data_to_process)

    # 绘制分解后的信号并保存图像
    plt.figure(figsize=(8, 6))
    plt.plot(data_to_process, 'r')
    plt.title("Original signal")
    plt.savefig(f"{target_name}_img/original_signal.png")
    plt.close()

    for idx, imf in enumerate(IMF):
        plt.figure(figsize=(8, 6))
        plt.plot(imf, 'g')
        plt.title("IMF " + str(idx + 1))
        plt.savefig(f"{target_name}_img/imf_" + str(idx + 1) + ".png")
        plt.close()
