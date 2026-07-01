import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import ipdb

class SplitedViT(nn.Module):
    def __init__(self, vit, splited_idx=5):
        super().__init__()
        self.vit = vit
        self.splited_idx = splited_idx
        self.device_blocks = self.vit.transformer.layers[0:splited_idx]
        self.cloud_server_blocks = self.vit.transformer.layers[splited_idx:]
        
        self.devices_layers = nn.ModuleList([
         self.vit.to_patch_embedding, 
            self.vit.dropout,
        ])
        for block in self.device_blocks:
            self.devices_layers.append(block)
            
        self.cloud_server_layers = nn.ModuleList([])
        for block in self.cloud_server_blocks:
            self.cloud_server_layers.append(block)
        self.cloud_server_layers.extend([
            self.vit.to_latent, 
            self.vit.mlp_head,
        ])

        
    def device_model_forward(self, img, mask=None, lora=None):
        if lora is not None:
            lora.register_lora(self.devices_layers)
            
        x = self.preprocessing(img)
        for attn, ff in self.device_blocks:
            x = attn(x, mask = mask)
            x = ff(x)
            
        if lora is not None:
            lora.remove_lora(self.devices_layers)
        return x
        
    def cloud_server_model_forward(self, x, mask=None, lora=None):
        if lora is not None:
            lora.register_lora(self.cloud_server_layers)
        for attn, ff in self.cloud_server_blocks:
            x = attn(x, mask = mask)
            x = ff(x)
        x = self.postprocessing(x)
        if lora is not None:
            lora.remove_lora(self.cloud_server_layers)
        return x
        
    def preprocessing(self, img):
        # img[b, c, img_h, img_h] > patches[b, p_h*p_w, dim]
        x = self.vit.to_patch_embedding(img)
        x = x.flatten(2).transpose(1,2)
        # ipdb.set_trace()
        b, n, _ = x.shape

        # cls_token[1, p_n*p_n*c] > cls_tokens[b, p_n*p_n*c]
        cls_tokens = repeat(self.vit.cls_token.to(img.device), '() n d -> b n d', b = b)
        # add(concat) cls_token to patch_embedding
        x = torch.cat((cls_tokens, x), dim=1)
        # add pos_embedding
        x += self.vit.pos_embedding[:, :(n + 1)].to(img.device)
        # drop out
        x = self.vit.dropout(x)
        return x

    def postprocessing(self, x):
        # use cls_token to get classification message
        x = x.mean(dim = 1) if self.vit.pool == 'mean' else x[:, 0]

        x = self.vit.to_latent(x)
        return self.vit.mlp_head(x)

    def forward(self, img, mask=None, lora=None):
        if lora is not None:
            lora.register_lora(self.vit)
        x = self.device_model_forward(img, mask=mask)
        x = self.cloud_server_model_forward(x, mask=mask)
        if lora is not None:
            lora.remove_lora(self.vit)
        return x