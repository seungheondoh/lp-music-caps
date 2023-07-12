import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn import metrics


def print_model_params(model):
    n_parameters = sum(p.numel() for p in model.parameters())
    train_n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("============")
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    print('number train of params (M): %.2f' % (train_n_parameters / 1.e6))
    print("============")
    
def load_pretrained(args, save_dir, model, model_types="last", mdp=False):
    pretrained_object = torch.load(f'{save_dir}/{model_types}.pth', map_location='cpu')
    state_dict = pretrained_object['state_dict']
    save_epoch = pretrained_object['epoch']
    if mdp:
        for k in list(state_dict.keys()):
            if k.startswith('module.'):
                state_dict[k[len("module."):]] = state_dict[k]
            del state_dict[k]
    model.load_state_dict(state_dict)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    model.eval()
    return model, save_epoch