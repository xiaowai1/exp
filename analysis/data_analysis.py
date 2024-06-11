# -*- coding = utf-8 -*-
# @Time : 2024/5/23 11:18
# @Author : ChiXiaoWai
# @File : data_analysis.py
# @Project : exp
# @Description : 初始数据分析
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
matplotlib.use('TkAgg')

# 定义集群名称
clusters = ['gc19_a']
# clusters = ['gc19_a', 'gc19_b', 'gc19_c', 'gc19_d', 'gc19_e', 'gc19_f', 'gc19_g', 'gc19_h']
# 生成文件路径
files = ['../data/' + cluster + '.csv' for cluster in clusters]
# "Init", "Frequency"
type = "Init"

# 初始数据分析
if type == "Init":
    # 创建一个图表用于绘制 cpu 列
    plt.figure(figsize=(15, 10))
    # 遍历每个文件，读取数据并绘制 cpu 列到同一张图表中
    for file in files:
        # 读取数据
        data = pd.read_csv(file, index_col=['date'], parse_dates=['date']).iloc[:2880]
        data.index = pd.to_datetime(data.index, unit='us')

        # 检查并绘制 cpu 列
        if 'avgcpu' in data.columns:
            plt.plot(data.index, data['avgcpu'], label=file.split('/')[-1])
        else:
            print(f"Column 'avgcpu' not found in {file}")

    # 添加图表标题和标签
    plt.title('CPU Usage from Multiple Files')
    plt.xlabel('Date')
    plt.ylabel('CPU Usage')
    # 显示图例
    plt.legend()
    # 保存图像
    plt.savefig('combined_cpu_data.png')

    # 创建一个图表用于绘制 mem 列
    plt.figure(figsize=(15, 10))
    # 遍历每个文件，读取数据并绘制 mem 列到同一张图表中
    for file in files:
        # 读取数据
        data = pd.read_csv(file, index_col=['date'], parse_dates=['date']).iloc[:2880]
        data.index = pd.to_datetime(data.index, unit='us')

        # 检查并绘制 mem 列
        if 'avgmem' in data.columns:
            plt.plot(data.index, data['avgmem'], label=file.split('/')[-1])
        else:
            print(f"Column 'avgmem' not found in {file}")

    # 添加图表标题和标签
    plt.title('Memory Usage from Multiple Files')
    plt.xlabel('Date')
    plt.ylabel('Memory Usage')
    # 显示图例
    plt.legend()
    # 保存图像
    plt.savefig('combined_mem_data.png')

# 频率分布图
elif type == "Frequency":
    # 创建图表，用于绘制 CPU 使用率的直方图
    plt.figure(figsize=(15, 7))
    # 遍历每个文件，绘制 CPU 使用率的直方图
    for idx, file in enumerate(files):
        # 读取数据
        data = pd.read_csv(file, index_col=['date'], parse_dates=['date'])
        data.index = pd.to_datetime(data.index, unit='us')
        # 提取 CPU 使用率数据
        cpu_data = data['avgcpu'].dropna().values
        # 绘制 CPU 使用率的直方图
        plt.hist(cpu_data, bins=150, alpha=0.7, edgecolor='black', label=file.split('/')[-1])
    # 添加标题和标签
    plt.title('CPU Usage Distribution')
    plt.xlabel('CPU Usage')
    plt.ylabel('Frequency')
    plt.legend()
    # 保存图像
    plt.savefig('cpu_distribution.png')


    # 创建图表，用于绘制内存使用率的直方图
    plt.figure(figsize=(15, 7))
    # 遍历每个文件，绘制内存使用率的直方图
    for idx, file in enumerate(files):
        # 读取数据
        data = pd.read_csv(file, index_col=['date'], parse_dates=['date'])
        data.index = pd.to_datetime(data.index, unit='us')
        # 提取内存使用率数据
        mem_data = data['avgmem'].dropna().values
        # 绘制内存使用率的直方图
        plt.hist(mem_data, bins=150, alpha=0.7, edgecolor='black', label=file.split('/')[-1])
    # 添加标题和标签
    plt.title('Memory Usage Distribution')
    plt.xlabel('Memory Usage')
    plt.ylabel('Frequency')
    plt.legend()
    # 保存图像
    plt.savefig('mem_distribution.png')
