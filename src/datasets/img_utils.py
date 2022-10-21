# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import os
import numpy as np
import cv2
from operator import attrgetter

def load(filepath, mode=None):
    if not os.path.exists(filepath):
        raise IOError("No such file or directory: '{}'".format(filepath))

    if mode is None:
        mode = cv2.IMREAD_UNCHANGED
    img = cv2.imread(filepath, mode)

    if img is None:
        print("ERROR Reading Image. Possibly corrupted PGM File:")
        print(filepath)
    if img.ndim > 2:
        if img.shape[-1] == 4:
            color_mode = cv2.COLOR_BGRA2RGBA
        else:
            color_mode = cv2.COLOR_BGR2RGB

        img = cv2.cvtColor(img, color_mode)
    return img


def save(filepath, img):
    img = np.asanyarray(img)

    if img.ndim == 2:
        cv2.imwrite(filepath, img)
    else:
        if img.shape[-1] == 4:
            color_mode = cv2.COLOR_RGBA2BGRA
        else:
            color_mode = cv2.COLOR_RGB2BGR
        if not cv2.imwrite(filepath, cv2.cvtColor(img, color_mode)):
            dirname = os.path.dirname(filepath)
            if not os.path.exists(dirname):
                msg = "No such directory: '{}'".format(dirname)
            else:
                msg = "Cannot write image to '{}'".format(filepath)
            raise IOError(msg)


def _rint(value):
    """Round and convert to int"""
    return int(np.round(value))

def _const(*args):
    """
    Return constant depending on OpenCV version.
    Returns first value found for supplied names of constant.
    """
    for const in args:
        try:
            return attrgetter(const)(cv2)
        except AttributeError:
            continue
    raise AttributeError(
        """Installed OpenCV version {:s} has non of the given constants.
         Tested constants: {:s}""".format(cv2.__version__, ', '.join(args))
    )


# interpolation modes (all supported)
_INTERPOLATION_DICT = {
    # bicubic interpolation
    'bicubic': _const('INTER_CUBIC', 'INTER_CUBIC'),
    # nearest-neighbor interpolation
    'nearest': _const('INTER_NEAREST', 'INTER_NEAREST'),
    # bilinear interpolation (4x4 pixel neighborhood)
    'linear': _const('INTER_LINEAR', 'INTER_LINEAR'),
    # resampling using pixel area relation, preferred for shrinking
    'area': _const('INTER_AREA', 'INTER_AREA'),
    # Lanczos interpolation (8x8 pixel neighborhood)
    'lanczos4': _const('INTER_LANCZOS4', 'INTER_LANCZOS4')
}

def dimshuffle(input_img, from_axes, to_axes):
    # check axes parameter
    if from_axes.find('0') == -1 or from_axes.find('1') == -1:
        raise ValueError("`from_axes` must contain both axis0 ('0') and"
                         "axis 1 ('1')")
    if to_axes.find('0') == -1 or to_axes.find('1') == -1:
        raise ValueError("`to_axes` must contain both axis0 ('0') and"
                         "axis 1 ('1')")
    if len(from_axes) != len(input_img.shape):
        raise ValueError("Number of axis given by `from_axes` does not match "
                         "the number of axis in `input_img`")

    # handle special cases for channel axis
    to_axes_c = to_axes.find('c')
    from_axes_c = from_axes.find('c')
    # remove channel axis (only grayscale image)
    if to_axes_c == -1 and from_axes_c >= 0:
        if input_img.shape[from_axes_c] != 1:
            raise ValueError('Cannot remove channel axis because size is not '
                             'equal to 1')
        input_img = input_img.squeeze(axis=from_axes_c)
        from_axes = from_axes.replace('c', '')

    # handle special cases for batch axis
    to_axes_b = to_axes.find('b')
    from_axes_b = from_axes.find('b')
    # remove batch axis
    if to_axes_b == -1 and from_axes_b >= 0:
        if input_img.shape[from_axes_b] != 1:
            raise ValueError('Cannot remove batch axis because size is not '
                             'equal to 1')
        input_img = input_img.squeeze(axis=from_axes_b)
        from_axes = from_axes.replace('b', '')

    # add new batch axis (in front)
    if to_axes_b >= 0 and from_axes_b == -1:
        input_img = input_img[np.newaxis]
        from_axes = 'b' + from_axes

    # add new channel axis (in front)
    if to_axes_c >= 0 and from_axes_c == -1:
        input_img = input_img[np.newaxis]
        from_axes = 'c' + from_axes

    return np.transpose(input_img, [from_axes.find(a) for a in to_axes])

def resize(img, shape_or_scale, interpolation='linear'):
    """
    Function to resize a given image.

    Parameters
    ----------
    img : {numpy.ndarray, list, tuple}
        The image to convert with axes either '01' or '01c' and of dtype
        'uint8', 'uint16' or 'float32'.
    shape_or_scale : {float, tuple, list}
        The output image shape as a tuple of ints (height, width), the scale
        factors for both dimensions as a tuple of floats (fy, fx) or a single
        float as scale factor for both dimensions.
    interpolation : str
        Interpolation method to use, one of: 'nearest', 'linear' (default),
        'area', 'bicubic' or 'lanczos4'. For details, see OpenCV documentation.

    Returns
    -------
    img_resized : numpy.ndarray
        The resized input image.

    """
    # ensure that img is a numpy object
    img = np.asanyarray(img)

    # get current shape
    cur_height, cur_width = img.shape[:2]

    # check shape_or_scale
    if isinstance(shape_or_scale, (tuple, list)) and len(shape_or_scale) == 2:
        if all(isinstance(e, int) for e in shape_or_scale):
            new_height, new_width = shape_or_scale
        elif all(isinstance(e, float) for e in shape_or_scale):
            fy, fx = shape_or_scale
            new_height = _rint(fy*cur_height)
            new_width = _rint(fx*cur_width)
        else:
            raise ValueError("`shape_or_scale` should either be a tuple of "
                             "ints (height, width) or a tuple of floats "
                             "(fy, fx)")
    elif isinstance(shape_or_scale, float):
        new_height = _rint(shape_or_scale * cur_height)
        new_width = _rint(shape_or_scale * cur_width)
    else:
        raise ValueError("`shape_or_scale` should either be a tuple of ints "
                         "(height, width) or a tuple of floats (fy, fx) or a "
                         "single float value")

    # scale image
    if cur_height == new_height and cur_width == new_width:
        return img

    return cv2.resize(img,
                      dsize=(new_width, new_height),
                      interpolation=_INTERPOLATION_DICT[interpolation])

