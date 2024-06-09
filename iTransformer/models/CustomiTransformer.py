# -*- coding = utf-8 -*-
# @Time : 2024/6/9 14:42
# @Author : ChiXiaoWai
# @File : CustomiTransformer.py
# @Project : exp
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from neuralforecast.common._modules import TransEncoder, TransEncoderLayer, AttentionLayer
from neuralforecast.models import iTransformer
from neuralforecast.models.informer import ProbAttention


class CustomiTransformer(iTransformer):
    def __init__(self, *args, **kwargs):
        super(CustomiTransformer, self).__init__(*args, **kwargs)
        self.encoder = TransEncoder(
            [
                TransEncoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            False, self.factor, attention_dropout=self.dropout
                        ),
                        self.hidden_size,
                        self.n_heads,
                    ),
                    self.hidden_size,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=F.gelu,
                )
                for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.hidden_size),
        )

    def forward(self, x):
        x = super(CustomiTransformer, self).forward(x)
        return x
