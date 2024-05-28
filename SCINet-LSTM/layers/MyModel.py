# -*- coding = utf-8 -*-
# @Time : 2024/5/28 9:48
# @Author : ChiXiaoWai
# @File : MyModel.py
# @Project : exp
import torch
import torch.nn as nn
from layers.Invertible import RevIN
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import Informer


# SCINet模型代码
class SCIBlock(nn.Module):
    def __init__(self, enc_in, kernel_size=3, dilation=1, dropout=0.5, d_model=64):
        super(SCIBlock, self).__init__()
        pad_l = dilation * (kernel_size - 1) // 2 + 1 if kernel_size % 2 != 0 else dilation * (kernel_size - 2) // 2 + 1
        pad_r = dilation * (kernel_size - 1) // 2 + 1 if kernel_size % 2 != 0 else dilation * (kernel_size) // 2 + 1

        self.phi = nn.Sequential(
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(enc_in, d_model, kernel_size=kernel_size, dilation=dilation),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, enc_in, kernel_size=3),
            nn.Tanh()
        )
        self.psi = nn.Sequential(
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(enc_in, d_model, kernel_size=kernel_size, dilation=dilation),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, enc_in, kernel_size=3),
            nn.Tanh()
        )
        self.rho = nn.Sequential(
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(enc_in, d_model, kernel_size=kernel_size, dilation=dilation),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, enc_in, kernel_size=3),
            nn.Tanh()
        )
        self.eta = nn.Sequential(
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(enc_in, d_model, kernel_size=kernel_size, dilation=dilation),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, enc_in, kernel_size=3),
            nn.Tanh()
        )

    def forward(self, x):
        x_even = x[:, ::2, :].transpose(1, 2)
        x_odd = x[:, 1::2, :].transpose(1, 2)

        x_odd_s = x_odd.mul(torch.exp(self.phi(x_even)))
        x_even_s = x_even.mul(torch.exp(self.psi(x_odd)))

        x_even_update = x_even_s + self.eta(x_odd_s)
        x_odd_update = x_odd_s - self.rho(x_even_s)

        return x_even_update.transpose(1, 2), x_odd_update.transpose(1, 2)


class SCITree(nn.Module):
    def __init__(self, level, enc_in, kernel_size=3, dilation=1, dropout=0.5, d_model=64):
        super(SCITree, self).__init__()
        self.level = level
        self.block = SCIBlock(
            enc_in=enc_in,
            kernel_size=kernel_size,
            dropout=dropout,
            dilation=dilation,
            d_model=d_model,
        )
        if level != 0:
            self.SCINet_odd = SCITree(level - 1, enc_in, kernel_size, dilation, dropout, d_model)
            self.SCINet_even = SCITree(level - 1, enc_in, kernel_size, dilation, dropout, d_model)

    def zip_up_the_pants(self, shape, even, odd):
        assert even.shape[1] == odd.shape[1]

        merge = torch.zeros(shape, device=even.device)
        merge[:, 0::2, :] = even
        merge[:, 1::2, :] = odd

        return merge  # [B, L, D]

    def forward(self, x):
        # [B, L, D]
        x_even_update, x_odd_update = self.block(x)

        if self.level == 0:
            return self.zip_up_the_pants(x.shape, x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(x.shape, self.SCINet_even(x_even_update), self.SCINet_odd(x_odd_update))


# 混合模型代码
class HybridModel(nn.Module):
    def __init__(self, configs):
        super(HybridModel, self).__init__()
        self.scinet_encoder = SCITree(level=3, enc_in=configs.enc_in, kernel_size=3, dilation=1, dropout=0.5,
                                      d_model=configs.d_model)
        self.informer = Informer(
            h=configs.pred_len,
            input_size=configs.seq_len,
            hidden_size=configs.d_model,
            dropout=configs.dropout,
            learning_rate=configs.learning_rate,
            loss=DistributionLoss(distribution='Normal', level=[80, 90]),
            hist_exog_list=None,  # 如果希望使用外部特征作为历史输入，请设置为相应的特征列表
            stat_exog_list=None,  # 如果希望使用外部特征作为静态输入，请设置为相应的特征列表
            futr_exog_list=None,  # 如果希望使用外部特征作为历史输入，请设置为相应的特征列表
        )

        self.projection = nn.Conv1d(configs.seq_len, configs.pred_len, kernel_size=1, stride=1, bias=False)
        self.rev = RevIN(configs.enc_in) if configs.rev else None

    def forward(self, x):
        x = self.rev(x, 'norm') if self.rev else x
        res = x
        # SCINet encoder
        scinet_encoded = self.scinet_encoder(x)
        scinet_encoded += res
        scinet_encoded = self.projection(scinet_encoded)

        atch_size, seq_len, feature_dim = scinet_encoded.size()
        scinet_encoded_flat = scinet_encoded.view(x.size(0) * seq_len, feature_dim)

        # Informer input dictionary
        informer_input = {'insample_y': scinet_encoded_flat, 'futr_exog': None}

        # Informer
        informer_output = self.informer.forward(informer_input)
        return self.rev(informer_output, 'denorm') if self.rev else informer_output
