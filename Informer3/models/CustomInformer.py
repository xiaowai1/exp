# -*- coding = utf-8 -*-
# @Time : 2024/6/8 13:13
# @Author : ChiXiaoWai
# @File : CustomInformer.py
# @Project : exp
import pandas as pd
import numpy as np
import pywt
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from neuralforecast.models import Informer
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast import NeuralForecast


# 自定义Informer模型
class CustomInformer(Informer):
    def __init__(self, *args, **kwargs):
        super(CustomInformer, self).__init__(*args, **kwargs)
        hidden_size = kwargs.pop('hidden_size', None)
        num_layers = 2
        # Custom layers or operations
        self.dsw_embedding = DSWEmbedding(self.input_size, hidden_size)
        self.tsa_layer = TSALayer(hidden_size, num_layers)

    def forward(self, x):
        # y为原先的x，x为原先x中的['insample_y']
        # y = x['insample_y']
        # y = self.dsw_embedding(y)
        # y = self.tsa_layer(y)
        # x['insample_y'] = y
        x = super(CustomInformer, self).forward(x)
        return x


class DSWEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DSWEmbedding, self).__init__()
        self.embedding_layer = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        embedded_x = self.embedding_layer(x)
        return embedded_x


class TSALayer(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(TSALayer, self).__init__()
        self.attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(hidden_size, num_heads=4) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.attention_layers:
            x, _ = layer(x, x, x)
        return x
