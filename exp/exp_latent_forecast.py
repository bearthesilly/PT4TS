"""
Experiment class for PT_forecast_latent.

Extends Exp_Long_Term_Forecast to:
  1. Pass ground-truth future (y_true) and current_epoch to the model during training
  2. Add the model's auxiliary latent-consistency loss to the prediction loss
  3. Gradient clipping for training stability
  4. Log both components separately for debugging
"""

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")


class Exp_Latent_Forecast(Exp_Long_Term_Forecast):

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

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

        print(self.args.model, "training start!")
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            latent_loss_accum = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                gt_future = batch_y[:, -self.args.pred_len :, :]

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark,
                            y_true=gt_future,
                            current_epoch=epoch,
                        )
                        if isinstance(outputs, tuple):
                            outputs, aux_loss = outputs
                            aux_loss = aux_loss.mean()
                        else:
                            aux_loss = torch.tensor(0.0, device=self.device)

                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        target = batch_y[:, -self.args.pred_len :, f_dim:]
                        loss = criterion(outputs, target) + aux_loss
                        train_loss.append(loss.item())
                        latent_loss_accum.append(aux_loss.item())
                else:
                    outputs = self.model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark,
                        y_true=gt_future,
                        current_epoch=epoch,
                    )
                    if isinstance(outputs, tuple):
                        outputs, aux_loss = outputs
                        aux_loss = aux_loss.mean()
                    else:
                        aux_loss = torch.tensor(0.0, device=self.device)

                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    target = batch_y[:, -self.args.pred_len :, f_dim:]
                    loss = criterion(outputs, target) + aux_loss
                    train_loss.append(loss.item())
                    latent_loss_accum.append(aux_loss.item())

                if (i + 1) % 100 == 0:
                    avg_lat = np.mean(latent_loss_accum[-100:])
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} (latent: {3:.7f})".format(
                            i + 1, epoch + 1, loss.item(), avg_lat
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(model_optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            avg_latent = np.average(latent_loss_accum)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} (Latent: {3:.7f}) "
                "Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                    epoch + 1, train_steps, train_loss, avg_latent, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model
