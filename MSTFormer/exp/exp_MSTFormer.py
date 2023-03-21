# import resource
# import psutil
import log
from data.data_loader import Dataset_Custom
from exp.exp_basic import Exp_Basic
from models.model import MSTFormer
from models.myloss import MyLoss
import random
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, distance
from utils.timefeatures import time_features

import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
# torch.set_num_threads(1)
import log as L

warnings.filterwarnings('ignore')
import datetime


class Exp_MSTFormer(Exp_Basic):
    def __init__(self, args):
        super(Exp_MSTFormer, self).__init__(args)
        self.lossfunction = MyLoss()
        currentDT = datetime.datetime.now()
        self.logger = currentDT.strftime("logs/%Y-%m-%d-%H-%M-%S") + ".txt"
        # with open(self.logger, "w") as f:
        #     pass

    def _build_model(self):
        model_dict = {
            'MSTFormer': MSTFormer,
        }
        if self.args.model == 'MSTFormer':
            e_layers = self.args.e_layers
            # print(e_layers)
            # L.write_log(e_layers)
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,  # start token length of MSTFormer decoder
                self.args.pred_len,
                self.args.prob_use,
                self.args.factor,  # probsparse attn factor
                self.args.d_model,  # dimension of model
                self.args.n_heads,
                e_layers,  # self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,  # dimension of fcn
                self.args.dropout,
                self.args.attn,  # attention used in encoder, options:[prob, full]
                self.args.embed,  # time features encoding
                self.args.freq,  # 分钟级，秒级等
                self.args.activation,
                self.args.output_attention,  # whether to output attention in ecoder
                # whether to use distilling in encoder, using this argument means not using distilling
                self.args.distil,
                self.args.mix,  # use mix attention in generative decoder
                self.device
            ).float()

        # 如果使用GPU，根据数目，并行处理
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

        # 获取数据并进行处理，返回符合输入格式的数据

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ship': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        # 如果embed为'timeF'，编码时间为0
        timeenc = 0 if args.embed != 'timeF' else 1
        # 随机打散数据shuffle_flag
        if flag == 'test':
            shuffle_flag = False;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
        # 做数据时，train和vali的选项
        else:
            # 更改为不打乱数据
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            data_type=args.data_type,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            prob_use=args.prob_use,
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols,
            read_processed=args.read_processed,
            logger=self.logger
        )
        s = str(flag) + str(len(data_set))
        L.write_log(s)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)  # 参数：待优化的参数集合，lr：学习率
        return model_optim

    def _select_criterion(self):
        criterion = self.lossfunction
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        # 作用等同于 self.train(False)简而言之，就是评估模式。而非训练模式。
        # 在评估模式下，batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移。
        self.model.eval()
        total_loss = []
        # 针对评估数据进行评估
        for i, (batch_x, batch_y, batch_x_map, batch_y_map, batch_x_mark, batch_y_mark, batch_orig) in enumerate(
                vali_loader):
            pred, true = self.process_one_batch(
                vali_data, batch_x, batch_y, batch_x_map, batch_y_map, batch_x_mark, batch_y_mark, batch_orig)
            # loss,_ = criterion(pred[:, :, :2].detach().to(self.device), true[:, :, :].detach().to(self.device))
            loss,_ = criterion(pred[:, :, :2].detach().to(self.device), true[:, :, :].detach().to(self.device))
            # .detach()返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,
            # 不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。
            # （我认为指当进行评估时，输出是什么就是什么，不应该将loss反传回去）
            total_loss.append(loss.cpu())
        total_loss = np.average(total_loss)
        self.model.train()  # 启用 BatchNormalization 和 Dropout,self.model.eval()不启用这两，评估完进行恢复
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')  # 所有数据，分割encoder，decoder后数据
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)  # location of model checkpoints
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)  # early stopping patience

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:  # use automatic mixed precision training
            # 如果特定op的正向传递具有浮点16输入，则该op的反向传递将产生浮点16梯度。具有小幅度的梯度值可能无法在浮点16中表示。
            # 这些值将刷新为零（“下溢”），因此相应参数的更新将丢失。为了防止下溢，"梯度缩放"将网络的损失（es）乘以比例因子，
            # 并调用缩放损失（es）的反向传递。然后，通过网络向后流动的梯度将按相同的系数缩放。换句话说，梯度值的幅度更大，因此它们不会刷新为零。
            # 在优化器更新参数之前，每个参数的梯度（.grad属性）都应取消缩放，因此缩放因子不会干扰学习速率。
            scaler = torch.cuda.amp.GradScaler()

        # 根据数据长度确定训练多少步
        self.args.train_epochs = len(train_loader)
        for epoch in range(self.args.train_epochs):  # set to 20
            iter_count = 0
            train_loss = []

            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_map, batch_y_map, batch_x_mark, batch_y_mark, batch_orig) in enumerate(
                    train_loader):
                iter_count += 1
                model_optim.zero_grad()

                # print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                # print(u'内存占比：', psutil.virtual_memory().percent)
                pred, true = self.process_one_batch(
                    train_data, batch_x, batch_y, batch_x_map, batch_y_map, batch_x_mark, batch_y_mark, batch_orig)

                # print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                # 只对经纬度做loss
                # loss,_ = criterion(pred[:, :, :2], true[:, :, :])
                loss,_ = criterion(pred[:, :, :2], true[:, :, :])
                train_loss.append(loss.item())
                # if iter_count<3:
                #     train_loss=(train_loss+loss.item())/2
                # else:
                #     train_loss = (train_loss*(iter_count-1) + loss.item()) / iter_count

                # print(u'内存占比：', psutil.virtual_memory().percent)
                # print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

                # del batch_x, batch_y, batch_x_map, batch_y_map, batch_x_mark, batch_y_mark, pred, true
                # torch.cuda.empty_cache()

                # 输出迭代次数，还需时间，loss，epoch等
                # if (i+1) % 100==0:  #迭代次数低于100
                # if i < 100:  # 迭代次数低于100
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

                # use automatic mixed precision training
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    # print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
                    model_optim.step()
                    # print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            # 反向传播后的时间
            s = "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time)
            L.write_log(s)
            # print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            s = "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss)
            L.write_log(s)
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                s = "Early stopping"
                L.write_log(s)
                # print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        criterion = self._select_criterion()

        self.model.eval()

        preds = []
        trues = []
        trajectorys = []

        for i, (batch_x, batch_y, batch_x_map, batch_y_map, batch_x_mark, batch_y_mark, batch_orig) in enumerate(
                test_loader):
            pred, true = self.process_one_batch(
                test_data, batch_x, batch_y, batch_x_map, batch_y_map, batch_x_mark, batch_y_mark, batch_orig)
            preds.extend(pred[:, :, :2].detach().cpu().numpy())
            trues.extend(true[:, :, :].detach().cpu().numpy())
            trajectorys.extend(batch_orig[:, :, :].detach().cpu().numpy())
        preds = np.array(preds)
        trues = np.array(trues)
        trajectorys = np.array(trajectorys)
        # print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        trajectorys = trajectorys.reshape(-1, trajectorys.shape[-2], trajectorys.shape[-1])
        s = 'test shape:' + str(preds.shape) + str(trues.shape)
        L.write_log(s)
        # print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        dis, preds = criterion(torch.tensor(preds[:, :, :2]), torch.tensor(trues[:, :, :]))
        mae, mse, rmse, mape, mspe = metric(np.array(preds), np.array(trues[:, 1:, :2]))
        # dis = distance(preds[:, :, 0],preds[:, :, 1],trues[:, :, 0],trues[:, :, 1])
        s = 'mse:{}, mae:{}, dis:{}, rmse{}, mape{}, mspe{}'.format(mse, mae, dis, rmse, mape, mspe)
        # s = 'mse:{}, mae:{}'.format(mse, mae)
        L.write_log(s)
        # print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'trajectory.npy', trajectorys)

        return dis

    # 处理一个batch
    def process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_map, batch_y_map, batch_x_mark,
                              batch_y_mark, batch_orig):
        batch_x = batch_x.float().to(self.device)  # 32*96*5
        batch_y = batch_y.float()  # 32,72,5

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        batch_x_map = batch_x_map.float().to(self.device)
        batch_y_map = batch_y_map.float().to(self.device)

        batch_orig = batch_orig.float().to(self.device)

        # decoder input
        # padding 0 或者 1
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()  # 24
        # 将两个张量（tensor）拼接在一起，cat是concatenate的意思
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)  # 48+24
        # encoder - decoder
        if self.args.use_amp:
            # 自动混合精度
            # Underflow（下溢出），在训练后期，例如激活函数的梯度会非常小， 甚至在梯度乘以学习率后，值会更加小。
            # （权重 = 旧权重 + lr * 梯度）
            # 即使了混合精度训练，还是存在无法收敛的情况，原因是激活梯度的值太小，造成了溢出。可以通过使用torch.cuda.amp.GradScaler，
            # 通过放大loss的值来防止梯度的underflow（只在BP时传递梯度信息使用，真正更新权重时还是要把放大的梯度再unscale回去）；
            # 反向传播前，将损失变化手动增大2^k倍，因此反向传播时得到的中间变量（激活函数梯度）则不会溢出；
            # 反向传播后，将权重梯度缩小2^k倍，恢复正常值
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, batch_x_map, batch_y_map, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, batch_x_map, batch_y_map, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, batch_x_map, batch_y_map, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, batch_x_map, batch_y_map, dec_inp, batch_y_mark)
        if self.args.inverse:
            # if data.shape[-1] != mean.shape[-1]:
            # mean = mean[-1:] std = std[-1:]
            outputs = dataset_object.inverse_transform(outputs)

        # self.args.features=='MS'，多特征预测一个特征，只预测最后一列
        f_dim = -1 if self.args.features == 'MS' else 0
        # true_time = batch_time[:, -(self.args.pred_len - 1):]
        # 三维，取预测长度设置的最后几个变量
        true = batch_orig[:, -(self.args.pred_len+1):, :].to(self.device)
        # batch_y = self.interpolation(true, true_time)
        # 轨迹预测，只预测前两列
        # batch_y = batch_y[:, -self.args.pred_len:, :2].to(self.device)
        # true=true.float().to(self.device)
        return outputs, true
