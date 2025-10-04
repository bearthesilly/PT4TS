import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer, PT, PT_forecast, PT_imputation, PT_anomaly, PT_implicit_HMM, PT_explicit_HMM, PT_explicit_HMM_without_G, \
    PT_forecast_v1, PT_forecast_v2, PT_forecast_v3, PT_forecast_v4, PT_forecast_v5, PT_forecast_v6, Transformer_ablation, PT_forecast_v7, PT_forecast_v8, \
    PT_forecast_v9, PT_forecast_v10, MLP, PT_forecast_v11, Transformer_vanilla, Transformer_exp, PT_forecast_v12, TSFusion, \
    Transformer_vanilla_implicit, Transformer_vanilla_explicit, PT_forecast_v13, PT_forecast_v14


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'WPMixer': WPMixer,
            'MultiPatchFormer': MultiPatchFormer,
            'PT': PT, # originally for classification
            'PT_forecast': PT_forecast,
            'PT_imputation': PT_imputation, 
            'PT_anomaly': PT_anomaly,
            'PT_implicit_HMM': PT_implicit_HMM,
            'PT_explicit_HMM': PT_explicit_HMM,
            'PT_explicit_HMM_without_G': PT_explicit_HMM_without_G,
            'PT_forecast_v1': PT_forecast_v1,
            'PT_forecast_v2': PT_forecast_v2, 
            'PT_forecast_v3': PT_forecast_v3,
            'PT_forecast_v4': PT_forecast_v4,
            'PT_forecast_v5': PT_forecast_v5,
            'PT_forecast_v6': PT_forecast_v6,
            'Transformer_ablation': Transformer_ablation,
            'PT_forecast_v7': PT_forecast_v7,
            'PT_forecast_v8': PT_forecast_v8,
            'PT_forecast_v9': PT_forecast_v9,
            'PT_forecast_v10': PT_forecast_v10,
            'PT_forecast_v11': PT_forecast_v11,
            'MLP': MLP,
            'Transformer_vanilla': Transformer_vanilla,
            'Transformer_exp': Transformer_exp,
            'PT_forecast_v12': PT_forecast_v12,
            'TSFusion': TSFusion,
            'Transformer_vanilla_implicit': Transformer_vanilla_implicit,
            'Transformer_vanilla_explicit': Transformer_vanilla_explicit,
            'PT_forecast_v13': PT_forecast_v13,
            'PT_forecast_v14': PT_forecast_v14
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"\nlearnable weight parameters: {trainable_params:,}")
        print(f"total parameters: {total_params:,}")
    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
