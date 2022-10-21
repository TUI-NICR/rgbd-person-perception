# -*- coding: utf-8 -*-
"""
.. codeauthor:: Tim Wengefeld <tim.wengefeld@tu-ilmenau.de>
"""

import torch
import torch.nn as nn
import numpy as np

class VonmisesLossBiternion(torch.nn.Module):
    """Von mises loss function for biternion inputs
    see: Beyer et al.: Biternion Nets: Continuous Head Pose Regression from
         Discrete Training Labels, GCPR 2015.
    """
    def __init__(self, kappa):
        super(VonmisesLossBiternion, self).__init__()
        self._kappa = kappa

    def forward(self, prediction, target):
        cos_angles = torch.cos(torch.deg2rad(prediction.float()) - torch.deg2rad(target.float()))
        cos_angles = torch.exp(self._kappa * (cos_angles - 1))
        score = 1 - cos_angles
        return score.masked_fill(target == -100, 0)

def biternion2deg(biternion):
    rad = torch.atan2(biternion[:, 1], biternion[:, 0])
    return torch.rad2deg(rad)


def biternion2deg_numpy(biternion):
    rad = np.arctan2(biternion[:, 1], biternion[:, 0])
    return np.rad2deg(rad)