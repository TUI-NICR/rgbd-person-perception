# -*- coding: utf-8 -*-
"""
.. codeauthor:: Tim Wengefeld <tim.wengefeld@tu-ilmenau.de>
"""
from models.losses import biternion2deg, biternion2deg_numpy
from models.model_utils import attribute_catergory
import torch
import numpy as np

def postprocess(nw_output):
    results = dict()
    softmax = torch.nn.Softmax(dim=1).to(device='cpu')
    for k, v in nw_output.items():
        if attribute_catergory[k] == 'binary':
            results[k] = softmax(v.to('cpu'))
        if attribute_catergory[k] == 'multiclass:3':
            results[k] = softmax(v.to('cpu'))
        if attribute_catergory[k] == 'biternion':
            results[k] = biternion2deg(v.to('cpu'))
    return results
