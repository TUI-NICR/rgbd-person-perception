# -*- coding: utf-8 -*-
"""
.. codeauthor:: Tim Wengefeld <tim.wengefeld@tu-ilmenau.de>
"""

import os
import time
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import torch


import datasets.img_utils as img_utils
from datasets.io_utils import get_files_by_extension
from datasets.preprocessing import preprocess_rgb, preprocess_depth
from utils.plot_helper import plot_sample
from utils.postprocess import postprocess

import onnx
import onnxruntime as ort

DEPTH_SUFFIX = "_depth.png"
RGB_SUFFIX = "_rgb.png"

def _parse_args():
    """Parse command-line arguments"""
    desc = 'Apply RGB-D neural network for attribute estimation'
    parser = ap.ArgumentParser(description=desc,
                               formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('onnx_filepath',
                        type=str,
                        help=("Path to onnx model file"))

    parser.add_argument('image_folderpath',
                        type=str,
                        help=("Path to a folder containing RGB and depth images with "
                              "_rgb.png and _depth.png suffix and corresponding filenames"))

    # other -------------------------------------------------------------------
    parser.add_argument('-c', '--cpu',
                        default=False,
                        action='store_true',
                        help="CPU only, do not run with GPU support")

    parser.add_argument('-p', '--profile',
                        default=False,
                        action='store_true',
                        help="Enable profiling")

    return parser.parse_args()

def main():
    # parse args --------------------------------------------------------------
    args = _parse_args()

    attributes = ['gender','has_long_trousers','has_jacket','has_long_sleeves','has_long_hair','is_person','orientation','posture']

    print()
    # load data --------------------------------------------------------------
    print("Load data from", args.image_folderpath)
    depth_filepaths = get_files_by_extension(args.image_folderpath, extension=DEPTH_SUFFIX.lower(), flat_structure=True, recursive=True, follow_links=True)
    print("Found",len(depth_filepaths),"depth patches in folder")

    # preprocess data --------------------------------------------------------------
    depth_images = []
    rgb_images = []
    for depth_path in depth_filepaths:
        if os.path.exists(str(depth_path).replace(DEPTH_SUFFIX,RGB_SUFFIX)):
            depth_images.append(img_utils.load(depth_path))
            rgb_images.append(img_utils.load(str(depth_path).replace(DEPTH_SUFFIX,RGB_SUFFIX)))

    print("found",len(rgb_images),"corresponding rgb patches in folder")
    print()
    batched_data_depth = np.empty((len(depth_images), 3, 224, 224), np.float32)
    batched_data_rgb = np.empty((len(rgb_images), 3, 224, 224), np.float32)
    for index, (depth, rgb) in enumerate(zip(depth_images,rgb_images)):
        batched_data_depth[index] = preprocess_depth(image=depth,output_size=(224,224))
        batched_data_rgb[index] = preprocess_rgb(image=rgb,output_size=(224,224))

    # Load ONNX model --------------------------------------------------------------
    if args.cpu:
        onnx_provider = 'CPUExecutionProvider'
    else:
        onnx_provider = 'CUDAExecutionProvider'
        
    opt = ort.SessionOptions()
    opt.enable_profiling = args.profile
    
    if args.profile:
        # see: https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Graph_Optimizations.md
        # ORT_DISABLE_ALL / ORT_ENABLE_BASIC / ORT_ENABLE_EXTENDED / ORT_ENABLE_ALL
        opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL    # default as well
        opt.intra_op_num_threads = 1    # only useful for cpu provider

        # enable logs
        opt.log_severity_level = 0   # -1

        # see: https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Perf_Tuning.md#profiling-and-performance-report
        # load resulting json file using chrome://tracing/ subsequently

    print('Load ONNX model',args.onnx_filepath)
    print()
    onnx_model = onnx.load(args.onnx_filepath)
    ort_session = ort.InferenceSession(
        onnx_model.SerializeToString(),
        providers=[onnx_provider], 
        sess_options=opt
    )
    
    # warmup
    onnx_preds = ort_session.run(None, {'rgb':batched_data_rgb,'depth':batched_data_depth})
    
    #actual inference
    start = time.time()
    onnx_preds = ort_session.run(None, {'rgb':batched_data_rgb,'depth':batched_data_depth})
    end = time.time()
    
    if args.profile:
        prof_file = ort_session.end_profiling()
    
    print('Inference took','{:.2f}'.format(end - start),'seconds with onnx provider',onnx_provider)
    print('This is','{:.2f}'.format(len(depth)/(end - start)),'patches per second')

    preds = {}
    preds['gender'] = torch.from_numpy(onnx_preds[0])
    preds['has_jacket'] = torch.from_numpy(onnx_preds[1])
    preds['has_long_sleeves'] = torch.from_numpy(onnx_preds[2])
    preds['has_long_hair'] = torch.from_numpy(onnx_preds[3])
    preds['has_long_trousers'] = torch.from_numpy(onnx_preds[4])
    preds['is_person'] = torch.from_numpy(onnx_preds[5])
    preds['orientation'] = torch.from_numpy(onnx_preds[6])
    preds['posture'] = torch.from_numpy(onnx_preds[7])

    results = postprocess(preds)
    plot_sample(rgb_images,depth_images,results)

if __name__ == '__main__':
    main()
