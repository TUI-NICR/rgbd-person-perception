# -*- coding: utf-8 -*-
"""
.. codeauthor:: Tim Wengefeld <tim.wengefeld@tu-ilmenau.de>
"""

import numpy as np
from torch import nn

class Identity(nn.Module):
    def __init__(self):
        """
        Used to remove the fc layer after the global average pooling from
        pretrained models.
        """
        super(Identity, self).__init__()

    def forward(self, x):
        return x

attribute_catergory = {
    'is_person':'binary',
    'gender':'binary',
    'has_jacket':'binary',
    'has_long_sleeves':'binary',
    'has_long_hair':'binary',
    'has_long_trousers':'binary',
    'has_glasses':'binary',
    'orientation':'biternion',
    'posture':'multiclass:3'
}

attribute_available_input = {
    'is_person': ['depth'],
    'gender': ['rgbd'],
    'has_jacket': ['rgbd'],
    'has_long_sleeves': ['rgbd'],
    'has_long_hair': ['rgbd'],
    'has_long_trousers': ['rgbd'],
    'has_glasses': ['rgbd'],
    'orientation':['depth'],
    'posture': ['depth']
}