import torch
import torch.nn as nn
from layers.Invertible import RevIN


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


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.encoder_ffn = nn.Sequential(
            nn.Linear(configs.seq_len * configs.enc_in, configs.d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(configs.dropout)
        )
        self.encoder = SCITree(level=1, enc_in=configs.enc_in, kernel_size=3, dilation=1, dropout=0.5,
                               d_model=configs.d_model)
        '''
        input_size: 输入特征的维度，即每个时间步输入张量的大小。
        hidden_size: 隐藏层的特征数量。它定义了LSTM单元输出的特征的维度。
        num_layers: LSTM堆叠的层数。多层LSTM可以增加模型的复杂度和能力。
        bias: 如果为True，则在LSTM单元中添加偏置。
        batch_first: 如果设置为True，则输入和输出张量的批处理维度(batch_size)将是第一维（形状为[batch_size, seq_len, feature]），否则第二维（默认情况下是[seq_len, batch_size, feature]）。
        dropout: 如果大于0，则在除最后一层外的每层后添加一个Dropout层。Dropout可以防止网络过拟合。
        bidirectional: 如果为True，则成为双向LSTM。双向LSTM可以从两个方向处理序列数据，通常能够提高模型性能。
        '''
        # self.lstm = nn.LSTM(input_size=configs.enc_in, hidden_size=configs.enc_in, num_layers=1,
        #                     dropout=configs.dropout, bidirectional=True, batch_first=True)
        # self.decoder = nn.Linear(configs.d_model * 2 * configs.seq_len, configs.pred_len)

        self.decoder_ffn = nn.Sequential(
            nn.Linear(configs.seq_len * configs.enc_in, configs.d_model),  # FFN层输入大小
            nn.ReLU(inplace=True),
            nn.Dropout(configs.dropout)
        )
        self.lstm = nn.LSTM(input_size=configs.d_model, hidden_size=configs.enc_in, num_layers=2,
                            dropout=configs.dropout, bidirectional=True, batch_first=True)

        self.projection = nn.Conv1d(configs.seq_len, configs.pred_len, kernel_size=1, stride=1, bias=False)
        self.rev = RevIN(configs.enc_in) if configs.rev else None

    def forward(self, x):
        x = self.rev(x, 'norm') if self.rev else x
        res = x
        x = self.encoder_ffn(x.contiguous().view(x.size(0), -1))
        x = x.view(x.size(0), -1, 2)
        x = self.encoder(x)
        x += res
        x = self.decoder_ffn(x.contiguous().view(x.size(0), -1))  # FFN层
        x = x.view(x.size(0), -1, 512)
        x, (h_n, c_n) = self.lstm(x)
        x = self.projection(x)  # 投影到输出大小
        x = self.rev(x, 'denorm') if self.rev else x
        return x
