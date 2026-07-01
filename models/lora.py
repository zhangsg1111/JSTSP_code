import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize # 需要版本高的torch
import math
import numpy as np
import warnings

# 先写一个lora模块
class LoRA_module(nn.Module):
    """
    only work for linear layer
    """
    def __init__(self, in_features, 
                 out_features, rank, 
                 register_layer, dtype, device, alpha=None):
        super(LoRA_module, self).__init__()
        self.alpha = rank if alpha==None else alpha
        self.scale = self.alpha/rank
        self.A = nn.Parameter(torch.randn(rank, in_features,
                                          dtype=dtype, device=device))
        self.B = nn.Parameter(torch.zeros(out_features, rank,
                                          dtype=dtype, device=device))# 原论文中只对B的参数置零
        self.register_layer = register_layer
        
    def forward(self, param):
        # 这个模块将使用于register_parametrization注册参数化
        # 所以这里输入的是模型参数，直接对这个模型参数param进行操作
        # return的是修改后的参数
        if len(param.size())==2: # linear layers
            return param + self.scale * self.B @ self.A
        if len(param.size())==4: # conv layers
            return param + (self.scale * self.B @ self.A).view(param.size())
        else:
            raise ValueError


class LoRA(nn.Module):
    def __init__(self, key_words, rank, alpha=None):
        """
            Attributes:
                model(nn.Module): 需要微调的模型
                key_words(dict of tuple): 需要微调的层的相关名字{'vae':('XXX'), 'unet':('XXX'),...}
        """
        super(LoRA, self).__init__()
        self.key_words = key_words
        self.rank = rank
        self.alpha = alpha
        self.modulelist = nn.ModuleList()
    
    def register_lora(self, nn_modules):
        init_lora = True if len(self.modulelist)==0 else False # 初始化lora参数
        m_names =  self.key_words
        modules_list = nn.ModuleList() if init_lora else self.modulelist
        idx = 0
        for name, module in nn_modules.named_modules():
            if name.endswith(m_names):
                # print(name)
                if init_lora is True:
                    out_dim = module.weight.shape[0]
                    in_dim = module.weight.shape[1]
                    if len(module.weight.shape)==2: # lora for linear layers
                        in_dim = in_dim
                    elif len(module.weight.shape)==4: # lora for conv layers
                        in_dim = in_dim*module.weight.shape[2]*module.weight.shape[3]

                    lora_module = LoRA_module(in_dim, out_dim, rank=self.rank, 
                                              register_layer = name,
                                              dtype=module.weight.dtype,
                                              device=module.weight.device, alpha=self.alpha)
                    modules_list.append(lora_module) # 将实例化的lora_module注册为这个LoRA的module
                else:
                    lora_module = modules_list[idx]
                    idx += 1
                if hasattr(module, 'parametrizations'): # 如果已经挂上一个lora，就先解绑它
                    warnings.warn('parametrizations already exist and is removed by default')
                    parametrize.remove_parametrizations(module, "weight", leave_parametrized=False)
                parametrize.register_parametrization(module, "weight", lora_module)
        if init_lora is True:
            self.modulelist = modules_list
        # print('lora was successfully registered into the model')

    def remove_lora(self, nn_modules):
        m_names = self.key_words
        for name, module in nn_modules.named_modules():
            if name.endswith(m_names):
                parametrize.remove_parametrizations(module, "weight", leave_parametrized=False)
        # print('lora was successfully removed from the model')