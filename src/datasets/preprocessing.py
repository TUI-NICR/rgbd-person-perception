# -*- coding: utf-8 -*-
"""
.. codeauthor:: Tim Wengefeld <tim.wengefeld@tu-ilmenau.de>
"""

from . import img_utils
import numpy as np


def rescale(image, output_size, image_type):
    if image_type=="depth":
        image = img_utils.resize(image, output_size, interpolation='nearest')
        return image[None,...] # append dim axis
    elif image_type=="rgb":
        image = img_utils.resize(image, output_size, interpolation='nearest')
        return image


def zeroMean(image, mean):
    image = image.astype(np.float32)
    image /= 255
    mean = np.array(mean, dtype=np.float32)
    image -= mean

    return image


def unitVariance(image, std):
    image = image.astype(np.float32)
    std = np.array(std, dtype=np.float32)
    image /= std

    return image


def scaleZeroOne(image, scale_factor):
    image = image.astype(np.float32)
    image /= float(scale_factor)

    return image


def preprocess_rgb(image, output_size):
    image = rescale(image, output_size,'rgb')

    #Werte von https://pytorch.org/hub/pytorch_vision_resnet/
    #image = ZeroMean(image, mean=[0.485, 0.456, 0.406])
    #image = UnitVariance(image, std=[0.229, 0.224, 0.225])

    # extracted from srl dataset RGB
    image = zeroMean(image, mean=[0.26236326, 0.25569668, 0.27034807])
    image = unitVariance(image, std=[0.14224607, 0.13232423, 0.13285837])

    # extracted from srl dataset YUV
    #mean [0.25935814 0.5071347  0.50429624]
    #std [0.13208283 0.01936206 0.03695451]

    image = img_utils.dimshuffle(image, '01c', 'c01')  

    return image

def preprocess_depth(image, output_size):
    image = rescale(image,output_size,'depth')
    image = scaleZeroOne(image,18000)
    #image -= 0.06111674
    #image /= 0.03162661
    image = np.tile(image, (3, 1, 1)) 
    
    return image
