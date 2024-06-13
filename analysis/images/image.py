# -*- coding = utf-8 -*-
# @Time : 2024/6/11 14:44
# @Author : ChiXiaoWai
# @File : image.py
# @Project : exp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
plt.switch_backend('agg')

# 读取数据
data = pd.read_csv("../../data/gc19_a.csv", index_col=['date'], parse_dates=['date']).iloc[:2880]

a_column_data = data['avgcpu'].iloc[100:160]
# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))
# 绘制折线图
ax.plot(a_column_data)
# 设置 y 轴范围为 0 到 1
ax.set_ylim(0, 1)
# 隐藏原有的 x 和 y 轴
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
# 设置箭头
arrowprops = dict(arrowstyle='->', color='black', linewidth=1.5)
# 添加 x 轴箭头
ax.annotate('', xy=(1, 0), xytext=(1.05, 0),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=arrowprops)

# 添加 y 轴箭头
ax.annotate('', xy=(0, 1), xytext=(0, 1.05),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=arrowprops)
# 禁用 x 和 y 轴的刻度显示
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
# plt.ylim(0, 1)
plt.savefig("./data.png")
