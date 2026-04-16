import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    AutoTimes-style rolling one-for-all baseline.

    It tokenizes each variable into fixed-length temporal patches, trains with
    causal next-patch prediction when y_true is supplied, and rolls out future
    patches autoregressively at inference time.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.patch_len = int(getattr(configs, "patch_len", 96))
        self.patch_len = max(1, self.patch_len)
        self.patch_num = math.ceil(self.seq_len / self.patch_len)
        self.future_patch_num = math.ceil(self.pred_len / self.patch_len)
        self.d_model = configs.d_model

        self.patch_embed = nn.Linear(self.patch_len, self.d_model)
        self.position_embed = nn.Parameter(
            torch.zeros(1, self.patch_num + self.future_patch_num + 8, self.d_model)
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=configs.n_heads,
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(enc_layer, num_layers=configs.e_layers)
        self.patch_head = nn.Linear(self.d_model, self.patch_len)

        nn.init.trunc_normal_(self.position_embed, std=0.02)

    def _normalize(self, x):
        means = x.mean(dim=1, keepdim=True).detach()
        x_centered = x - means
        stdev = torch.sqrt(torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return x_centered / stdev, means, stdev

    def _patchify(self, x, patch_num):
        # x: [B, T, K] -> [B, K, patch_num, patch_len]
        total_len = patch_num * self.patch_len
        if x.size(1) < total_len:
            x = F.pad(x, (0, 0, 0, total_len - x.size(1)))
        else:
            x = x[:, :total_len, :]
        return x.transpose(1, 2).reshape(x.size(0), self.enc_in, patch_num, self.patch_len)

    def _encode_patches(self, patches):
        # patches: [B, K, P, patch_len]
        batch_size, num_vars, patch_num, _ = patches.shape
        tokens = self.patch_embed(patches.reshape(batch_size * num_vars, patch_num, self.patch_len))
        tokens = tokens + self.position_embed[:, :patch_num, :]
        causal_mask = torch.triu(
            torch.full((patch_num, patch_num), float("-inf"), device=tokens.device),
            diagonal=1,
        )
        return self.backbone(tokens, mask=causal_mask)

    def _teacher_forced(self, x_norm, y_true_norm):
        hist_patches = self._patchify(x_norm, self.patch_num)
        future_patches = self._patchify(y_true_norm, self.future_patch_num)
        all_patches = torch.cat([hist_patches, future_patches], dim=2)
        encoded = self._encode_patches(all_patches)

        start = self.patch_num - 1
        stop = start + self.future_patch_num
        future_hidden = encoded[:, start:stop, :]
        pred_patches = self.patch_head(future_hidden)
        return pred_patches.view(x_norm.size(0), self.enc_in, self.future_patch_num, self.patch_len)

    def _rollout(self, x_norm):
        patches = self._patchify(x_norm, self.patch_num)
        pred_patches = []
        for _ in range(self.future_patch_num):
            encoded = self._encode_patches(patches)
            next_patch = self.patch_head(encoded[:, -1, :])
            next_patch = next_patch.view(x_norm.size(0), self.enc_in, 1, self.patch_len)
            pred_patches.append(next_patch)
            patches = torch.cat([patches, next_patch], dim=2)
        return torch.cat(pred_patches, dim=2)

    def forecast(self, x_enc, y_true=None):
        x_norm, means, stdev = self._normalize(x_enc)
        if self.training and y_true is not None:
            future = y_true[:, -self.pred_len:, :]
            y_true_norm = (future - means) / stdev
            pred_patches = self._teacher_forced(x_norm, y_true_norm)
        else:
            pred_patches = self._rollout(x_norm)

        out = pred_patches.reshape(x_enc.size(0), self.enc_in, -1)[:, :, :self.pred_len]
        out = out.permute(0, 2, 1)
        return out * stdev[:, 0:1, :] + means[:, 0:1, :]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, y_true=None, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            return self.forecast(x_enc, y_true=y_true)
        raise NotImplementedError("AutoTimes only supports forecasting tasks.")
