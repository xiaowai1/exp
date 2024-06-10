# -*- coding = utf-8 -*-
# @Time : 2024/5/16 20:45
# @Author : ChiXiaoWai
# @File : exp_informer.py
# @Project : code
from exp.exp_basic import ExpBasic
from utils.data_loader import DatasetCustom, DatasetPred
from utils.tools import EarlyStopping
from models.model import Informer, InformerStack
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from utils.metrics import metric
import os
import time


class ExpInformer(ExpBasic):
    def __init__(self, args):
        super(ExpInformer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack,
        }
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,  # self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, pre_data=None):
        args = self.args
        data_dict = {
            # 'ETTh1': Dataset_ETT_hour,
            # 'ETTh2': Dataset_ETT_hour,
            # 'ETTm1': Dataset_ETT_minute,
            # 'ETTm2': Dataset_ETT_minute,
            'WTH': DatasetCustom,
            'ECL': DatasetCustom,
            'Solar': DatasetCustom,
            'custom': DatasetCustom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = DatasetPred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # results save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, args, setting, load=False):
        # history_data = pd.read_csv(args.root_path + args.data_path)[args.target][-args.seq_len:].reset_index(drop=True)
        data_frames = []
        if args.is_rolling_predict:
            # 读取每个文件的数据框
            df = pd.read_csv(args.root_path + "/gc19_a.csv")
            # 计算要保留的后 20% 数据的起始索引
            start_index = int(0.8 * len(df))
            # 将读取的数据框的后 20% 数据追加到列表中
            data_frames.append(df.iloc[start_index:])
            pre_data = pd.concat(data_frames, ignore_index=True)
        else:
            pre_data = pd.read_csv(args.root_path + args.data_path)
        pre_data['date'] = pd.to_datetime(pre_data['date'], unit='us')
        columns = ['forecast' + column for column in pre_data.columns[1:]]
        pre_data.reset_index(inplace=True, drop=True)
        pre_length = args.pred_len
        # 数据都读取进来
        dict_of_lists = {column: [] for column in columns}
        results = []
        for i in range(int(len(pre_data) / pre_length)):
            if i == 0:
                pred_data, pred_loader = self._get_data(flag='pred')
            else:
                pred_data, pred_loader = self._get_data(flag='pred', pre_data=pre_data.iloc[:i * pre_length])

            print(f'预测第{i + 1} 次')
            if load:
                path = os.path.join(self.args.checkpoints, setting)
                best_model_path = path + '/' + 'checkpoint.pth'
                self.model.load_state_dict(torch.load(best_model_path))

            self.model.eval()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                pred, true = self._process_one_batch(
                    pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                # pred = pred_data.inverse_transform(pred)
                if args.features == 'MS' or args.features == 'S':
                    for i in range(args.pred_len):
                        results.append(pred[0][i][0].detach().cpu().numpy())
                else:
                    for j in range(args.enc_in):
                        for i in range(args.pred_len):
                            dict_of_lists[columns[j]].append(pred[0][i][j].detach().cpu().numpy())
                print("pred:", pred)
            if not args.is_rolling_predict:
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>不进行滚动预测<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                break

        if not args.is_rolling_predict:
            if args.features == 'MS' or args.features == 'S':
                df = pd.DataFrame({'forecast{}'.format(args.target): pre_data[args.target]})
                df.to_csv('Interval-{}'.format(args.data_path), index=False)
            else:
                df = pd.DataFrame(dict_of_lists)
                df.to_csv('Interval-{}'.format(args.data_path), index=False)
        else:
            if args.features == 'MS' or args.features == 'S':
                df = pd.DataFrame({'date': pre_data['date'], '{}'.format(args.target): pre_data[args.target],
                                   'forecast{}'.format(args.target): pre_data[args.target]})
                df.to_csv('Interval-{}'.format(args.data_path), index=False)
            else:
                df = pd.DataFrame(dict_of_lists)
                new_df = pd.concat((pre_data, df), axis=1)
                new_df.to_csv('Interval-{}'.format(args.data_path), index=False)
        pre_len = len(dict_of_lists['forecast' + args.target])
        # 绘图Avgmem
        matplotlib.use('TkAgg')
        plt.switch_backend('agg')

        for target in ['avgmem', 'avgcpu']:
            plt.figure()
            if args.is_rolling_predict:
                if args.features == 'MS' or args.features == 'S':
                    plt.plot(range(pre_len),
                             pre_data[target][:pre_len].tolist(), label='Actual Values')
                    plt.plot(range(pre_len), results,
                             label='Predicted Values')
                else:
                    plt.plot(range(pre_len),
                             pre_data[target][:pre_len].tolist(), label='Actual Values')
                    plt.plot(range(pre_len), dict_of_lists[f'forecast{target}'],
                             label='Predicted Values')
            else:
                if args.features == 'MS' or args.features == 'S':
                    plt.plot(range(len(results)), results,
                             label='Predicted Values')
                else:
                    plt.plot(range(len(dict_of_lists[f'forecast{target}'])),
                             dict_of_lists[f'forecast{target}'], label='Predicted Values')
            # 添加图例
            plt.legend()
            plt.style.use('ggplot')
            # 添加标题和轴标签
            plt.title(f'Past vs Predicted {target.upper()} Values')
            plt.xlabel('Time Point')
            plt.ylabel(target.upper())
            # 保存图表
            plt.savefig(f'{target.capitalize()}.png')
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y
