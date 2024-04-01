import os

import h5py
import numpy as np
from PIL import Image
import cv2
from utils.basic_utils import RunSteps, DataTypes,DepthTypes, DataTypesSUNRGBD
from utils.depth_utils import colorized_surfnorm


def cnn_or_rnn_features_loader(path):
    cnn_or_rnn_feats = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': [], 'layer5': [], 'layer6': [], 'layer7': []}
    img_file = h5py.File(path, 'r')
    for layer in cnn_or_rnn_feats.keys():
        cnn_or_rnn_feats[layer] = np.squeeze(np.asarray(img_file[layer]))

    return cnn_or_rnn_feats

def imread_uint(path: str, n_channels: int = 3) -> np.ndarray:
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # print(path)
        # print(img.shape)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise NotImplementedError
    return img
def custom_loader(path, params):
    #RGBD
    if params.data_type == DataTypes.RGBD:
        path1, path2 = path[0], path[1]

        #RGB image
        data_RGB = imread_uint(path1,n_channels=3)

        #colored Depth image
        if params.depth_type == DepthTypes.RGB:
            data_Depth = imread_uint(path2, n_channels=3)
        # original Depth image
        else:
            data_Depth = imread_uint(path2, n_channels=1)
            # results_dir = params.dataset_path + params.features_root + RunSteps.COLORIZED_DEPTH_SAVE + '/' + 'all_results_depthcrop'
            # if os.path.exists(results_dir):  # if colorized depth images are already saved read them
            #     img_path = results_dir + '/' + path2.split('/')[-1] + '.hdf5'
            #     img_file = h5py.File(img_path, 'r')
            #     data_type = 'colorized_depth'
            #     data_Depth = np.asarray(img_file[data_type])
            # else:
            #     img = colorized_surfnorm(path2)
            #     data_Depth = np.array(img, dtype=np.float32)

        # print(data_RGB.shape,data_Depth.shape)
        # cv2.imwrite('RGB.png', cv2.cvtColor(data_RGB, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('D.png',cv2.cvtColor(data_Depth, cv2.COLOR_RGB2BGR))
        return data_RGB, data_Depth
    #RGB
    elif params.data_type == DataTypes.RGB:
        path1 = path[0]
        data_RGB = imread_uint(path1, n_channels=3)
        return data_RGB
    #Depth
    else:
        path2 = path[0]
        if params.depth_type == DepthTypes.RGB:
            data_Depth = imread_uint(path2, n_channels=3)
        else:
            data_Depth = imread_uint(path2, n_channels=1)
        return data_Depth

def sunrgbd_loader(path, params):
    if params.data_type == DataTypesSUNRGBD.Depth:
        data_type = 'sunrgbd'
        img_file = h5py.File(path, 'r')
        return np.asarray(img_file[data_type], dtype=np.float32)
    else:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
