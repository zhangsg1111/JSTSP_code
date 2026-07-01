from typing import Tuple
import huffman
import math
import numpy as np
import torch
from torch import Tensor


class GolombPosEncoder:
    def __init__(self):
        self.phi = (np.sqrt(5) + 1) / 2
    
    @torch.no_grad()
    def encode(self, input: torch.Tensor, p: float):
        """
        For simulation only.
        Args:
            input (Tensor): the input binary tensor
            p (float): 1 - sparsity rate 
        """
        index = torch.nonzero(input.flatten(), as_tuple=True)[0]
        dist = torch.diff(index, prepend=torch.zeros(1, dtype=torch.long).to(input.device))
        
        bit_num = 1 + np.log2(np.log(self.phi - 1) / np.log(1 - p))
        q = dist.div(2**bit_num, rounding_mode='floor')
        return q.sum() + q.shape[0] * (1 + bit_num), index.shape[0]


def entropy_encode(freq_list):
    code_book = huffman.codebook([(str(i), fre) for i, fre in enumerate(freq_list)])
    # sum([freq_code_1*len_code_1, ..., freq_code_n*len_code_n])
    return sum([freq * len(code_book[str(i)]) for i, freq in enumerate(freq_list)])


class Compressor():
    def __init__(self, entropy_coding=True):
        self.sparsity = 0
        self.bit_width = 32
        self.entropy_coding = entropy_coding

    def set(self, sparsity, bit_width, entropy_coding=True):
        self.sparsity = sparsity
        self.bit_width = bit_width
        self.entropy_coding = entropy_coding

    @torch.no_grad()
    def compress(self, model_data):
        # sparsification
        num_param = model_data.numel()
        K = max(round((1-self.sparsity) * num_param), 1)
        ##### ori ####
        kth = max(round(self.sparsity * num_param), 1)
        flatten_abs_data = model_data.abs().view(-1)
        threshold_val = flatten_abs_data.kthvalue(kth).values.item()
        mask = flatten_abs_data >= threshold_val
        mask = mask.view(model_data.shape)
        sparse_data = model_data * mask
        num_param4quan = mask.sum().item()
        # quantization
        if self.bit_width!=32:
            quant_level = 2 ** (self.bit_width-1)
            sgn_data = torch.sgn(sparse_data)
            abs_data = torch.abs(sparse_data)
            data_list = abs_data[mask]
            values, indices = data_list.sort()
            min_val = values[0]
            max_val = values[-int(len(values)*0.05)] # truncate 5% maximum elements
            # max_val = values[-1]
            delta = (max_val - min_val) / (quant_level - 1)
            quant_points = torch.linspace(min_val, max_val, quant_level).to(model_data.device)

            gamma = (torch.rand_like(abs_data)+0.5) * delta/2
            quant_repr = torch.searchsorted(quant_points, abs_data+gamma)
            quant_repr[quant_repr>0] -= 1
            quant_data = min_val+quant_repr*delta
            quant_data[~mask] = 0 # pruned param. are not quantifized
            decompressed_data = quant_data * sgn_data
        else:
            quant_data = sparse_data
            decompressed_data = sparse_data
            self.entropy_coding = False
        # coding
        if self.entropy_coding is True:
            fre_array = quant_data[mask].float().detach().cpu().histogram(bins=quant_points.float().detach().cpu())[0]
            index_param_bit_num = entropy_encode(fre_array)
            size = index_param_bit_num + num_param4quan + num_param + 32*2 # quan_repr+ sign_mask+ pruning_mask+ min_val+ max_val (bit)
        else:
            size = num_param4quan * (self.bit_width) + num_param + 32*2 # (bit)
        return decompressed_data, size


