from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.args = args
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # Print the the component and the corresponding number of parameters
        if self.args.use_gpu:
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("Total trainable parameters: {:.2f}M".format(total_params / 1e6))
            print("Model components:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{name}: {param.numel()} parameters")
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    # Models that are pure statistical (no backprop needed)
    STATISTICAL_MODELS = {
        'VAR',
        'ARIMA',
        'BVAR',
        'GPVAR',
        'DynamicFactorModel',
        'StateSpaceModel',
    }

    TEACHER_FORCED_AR_MODELS = {
        'AutoTimes',
        'DeepVAR',
        'LSTM_AR',
        'TCN_AR',
    }

    def _model_forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y=None):
        if batch_y is not None and self.args.model in self.TEACHER_FORCED_AR_MODELS:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, y_true=batch_y)
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    def _probabilistic_loss(self, target):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(model, 'nll_loss'):
            return model.nll_loss(target)
        return None

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

        is_statistical = self.args.model in self.STATISTICAL_MODELS

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # Statistical models need only 1 epoch (they fit per-sample, no learning)
        num_epochs = 1 if is_statistical else self.args.train_epochs
        print(self.args.model, 'training start!')
        for epoch in range(num_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._model_forward(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y=batch_y
                        )

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = self._probabilistic_loss(batch_y)
                        if loss is None:
                            loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self._model_forward(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y=batch_y
                    )

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = self._probabilistic_loss(batch_y)
                    if loss is None:
                        loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                # if self.args.model == 'Transformer_vanilla':
                #     for j in range(self.args.e_layers):
                #         regularization_loss = torch.norm(self.model.encoder.layers[j].binary_potential_table, p=2)
                #         loss += 0.001*regularization_loss
                #     if (i + 1) % 100 == 0:
                #         print("Regulation Loss: {0:.7f}".format(regularization_loss.item()))
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                # Statistical models: no backprop needed
                if is_statistical:
                    continue
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

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        # if self.args.model == 'PT_forecast_v1' or self.args.model == 'PT_forecast_v8':
        #     self.model.model.iterator.head_selection.show_head_train()
        if self.args.model == 'PT_forecast_v8':
            self.model.model.iterator.head_selection.show_head_train()
        return self.model

    def test(self, setting, test=0):
        print("Start Testing!")
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # For testing channel sequence insensitivity 
        self.model.eval() 
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # For testing channel sequence insensitivity 
                # batch_x = batch_x.transpose(1, 2)[:, shuffle_indices, :].transpose(1, 2)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # outputs = outputs.transpose(1, 2)[:, reverse_indices, :].transpose(1, 2)
                # For testing channel sequence insensitivity 
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                if i % 20 == 0:
                    # 1. 准备数据
                    input_data = batch_x.detach().cpu().numpy()
                    
                    # 反归一化处理 (保持原有逻辑)
                    if test_data.scale and self.args.inverse:
                        shape = input_data.shape
                        input_data = test_data.inverse_transform(input_data.reshape(shape[0] * shape[1], -1)).reshape(shape)

                    # 2. 确定可视化通道数 (前10个 或 全部)
                    # data shape: [Batch, Seq_Len, Channels] -> 取 Channels 维度
                    num_channels = min(10, input_data.shape[2]) 
                    
                    # 3. 创建画布 (垂直排列 num_channels 个子图)
                    # figsize 宽12，高根据通道数自动拉长 (每个通道高2.5)
                    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2.5 * num_channels), sharex=True)
                    if num_channels == 1: axes = [axes] # 兼容单通道情况
                    
                    # 4. 循环绘制每个通道
                    for c in range(num_channels):
                        # 拼接: input历史 + ture/pred未来
                        # [0, :, c] 取当前batch第0号样本的第c个通道
                        gt_seq = np.concatenate((input_data[0, :, c], true[0, :, c]), axis=0)
                        pd_seq = np.concatenate((input_data[0, :, c], pred[0, :, c]), axis=0)
                        
                        # 绘图
                        axes[c].plot(gt_seq, label='GroundTruth', linewidth=1.5)
                        axes[c].plot(pd_seq, label='Prediction', linewidth=1.5, linestyle='--')
                        
                        # 标出预测起始线 (Seq_Len 处)
                        axes[c].axvline(x=input_data.shape[1], color='r', linestyle=':', alpha=0.5)
                        
                        axes[c].set_title(f'Channel {c}', fontsize=10)
                        axes[c].legend(loc='upper right', fontsize=8)
                    
                    plt.tight_layout()
                    
                    # 5. 保存并关闭
                    # 保持原有命名逻辑 str(i).pdf，但内容现在是多通道的了
                    plt.savefig(os.path.join(folder_path, str(i) + '.pdf'), bbox_inches='tight')
                    plt.close(fig) # 关键：释放内存，不显示
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            print("self.args.use_dtw is true")
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'
        print("Calculating metrics...")
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        per_sample_mse = ((preds - trues) ** 2).mean(axis=(1, 2))
        print(f'Per-sample MSE  | mean: {per_sample_mse.mean():.4f}  '
              f'median: {np.median(per_sample_mse):.4f}  '
              f'std: {per_sample_mse.std():.4f}  '
              f'min: {per_sample_mse.min():.4f}  '
              f'max: {per_sample_mse.max():.4f}  '
              f'sample-0: {per_sample_mse[0]:.4f}')

        fig_hist, ax_hist = plt.subplots(1, 1, figsize=(8, 4))
        ax_hist.hist(per_sample_mse, bins=min(50, len(per_sample_mse) // 2 + 1),
                     edgecolor='black', alpha=0.75)
        ax_hist.axvline(per_sample_mse.mean(), color='r', linestyle='--',
                        label=f'Mean = {per_sample_mse.mean():.4f}')
        ax_hist.axvline(per_sample_mse[0], color='g', linestyle=':',
                        label=f'Sample 0 = {per_sample_mse[0]:.4f}')
        ax_hist.set_xlabel('Per-sample MSE')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title(f'{self.args.model} — Per-sample MSE distribution')
        ax_hist.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, 'per_sample_mse_hist.pdf'),
                    bbox_inches='tight')
        plt.close(fig_hist)
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()
        # From xzz: stop storing them to save time and memory
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return
