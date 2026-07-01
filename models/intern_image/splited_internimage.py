import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class SplitedInternImage(nn.Module):
    def __init__(self, internimage, splited_idx=2):
        super().__init__()
        self.internimage = internimage
        self.splited_idx = splited_idx
        
        self.device_blocks = self.internimage.model.levels[0:splited_idx]
        self.cloud_server_blocks = self.internimage.model.levels[splited_idx:]
        
        self.devices_layers = nn.ModuleList([
             self.internimage.model.patch_embed, 
            self.internimage.model.pos_drop,
        ])
        for block in self.device_blocks:
            self.devices_layers.append(block)
            
        self.cloud_server_layers = nn.ModuleList([])
        for block in self.cloud_server_blocks:
            self.cloud_server_layers.append(block)
        self.cloud_server_layers.extend([
            self.internimage.model.dcnv3_head_x4, 
            self.internimage.model.dcnv3_head_x3,
            self.internimage.model.clip_projector,
            self.internimage.model.fc_norm,
        ])

        
    def device_model_forward(self, x, lora=None):
        if lora is not None:
            lora.register_lora(self.devices_layers)
            
        x = self.preprocessing(x)
        for level in self.device_blocks:
            x, x_ = level(x, return_wo_downsample=True)
        if lora is not None:
            lora.remove_lora(self.devices_layers)
        return x
        
    def cloud_server_model_forward(self, x, lora=None):
        if lora is not None:
            lora.register_lora(self.cloud_server_layers)
        seq_out = []
        for level in self.cloud_server_blocks:
            x, x_ = level(x, return_wo_downsample=True)
            seq_out.append(x_)
        x = self.postprocessing(seq_out)
        if lora is not None:
            lora.remove_lora(self.cloud_server_layers)
        return x
        
    def preprocessing(self, x):
        x = self.internimage.model.patch_embed(x)
        x = self.internimage.model.pos_drop(x)
        return x

    def postprocessing(self, xs):
        x3, x4 = xs
        
        x3 = x3.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x4 = x4.permute(0, 3, 1, 2)  # NHWC -> NCHW
        
        x4 = self.internimage.model.dcnv3_head_x4(x4)
        x = x4
        x3 = self.internimage.model.dcnv3_head_x3(x3)
        x = x + x3

        x = x.flatten(-2).transpose(1, 2).contiguous()
        x = self.internimage.model.clip_projector(x)
        x = self.internimage.model.fc_norm(x)
        logits = self.internimage.model.head(x)
        return logits

    def forward(self, x, lora=None):
        if lora is not None:
            lora.register_lora(self.internimage)
        x = self.device_model_forward(x)
        x = self.cloud_server_model_forward(x)
        if lora is not None:
            lora.remove_lora(self.internimage)
        return x