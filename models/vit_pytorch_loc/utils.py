import torch
import torch.nn as nn
import numpy as np
import math

def load_partial_weight(model, weight):
    model_dict = model.state_dict()
    state_dict = {k:v for k,v in weight.items() if k in model_dict.keys()}
    key_match_rate = len(state_dict)/len(weight)
    print('match the state_dict from loading weights, matching rate is ', key_match_rate*100, '%')
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)