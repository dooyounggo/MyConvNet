"""
Parameters for VGG16 demo.
Download ImageNet validation set at http://image-net.org/challenges/LSVRC/2012/downloads
Download checkpoint at https://github.com/tensorflow/models/tree/master/research/slim.
"""

import os
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from initialization import init_params
from dataset import DataSet
from subsets.ilsvrc_2012_cls import read_subset
from models.vggnet import VGG16 as ConvNet
from evaluators import AccuracyEvaluator as Evaluator
from models.init_from_checkpoint import vggnet as init_from_checkpoint
import utils


class Parameters(object):
    # FIXME: Directories
    _root_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # FIXME
    _train_dir = os.path.join(_root_dir, 'tfdatasets/ilsvrc_2012_cls/train')
    _train_sample_size = None  # Data size. None for all files in train_dir
    _val_dir = os.path.join(_root_dir, 'tfdatasets/ilsvrc_2012_cls/validation')  # Can be None
    _val_sample_size = None
    _test_dir = os.path.join(_root_dir, 'tfdatasets/ilsvrc_2012_cls/validation')
    _test_sample_size = 1000

    _save_dir = os.path.join(_root_dir, 'trained_models/VGG16_ImageNet')
    _transfer_dir = None
    _checkpoint_dir = os.path.join(_root_dir, 'pretrained_models', 'vgg_16', 'vgg_16.ckpt')

    d = dict()
    # FIXME: Image pre-processing hyperparameters
    d['image_size'] = (224, 224, 3)  # Processed image size
    d['image_size_test'] = (256, 256, 3)  # If None, same as 'image_size'
    d['resize_type'] = 'resize_expand'  # Resize types: 'resize', 'resize_expand', 'random_resized_crop', ...
    d['resize_type_test'] = None  # If None, same as 'resize_type'
    d['resize_random'] = True  # Randomness of padding and crop operations
    d['resize_random_test'] = False
    d['resize_interpolation'] = 'bicubic'  # Interpolation methods: 'nearest', 'bilinear', 'bicubic'

    d['input_size'] = (224, 224, 3)  # Network input size after augmentation
    d['image_mean'] = 0.5  # If None, it will be calculated and it may take some time
    d['zero_center'] = True  # Whether to zero-center the images
    d['shuffle'] = True  # Whether to shuffle the data

    # FIXME: Transfer learning parameters
    d['init_from_public_checkpoint'] = True  # Whether to use pre-trained model in checkpoint_dir

    def __init__(self, parser=None):
        print('Training directory: ', self.train_dir)
        print('Test directory: ', self.test_dir)
        print('Transfer learning directory: ', self.transfer_dir)
        print('Data save directory: ', self.save_dir)
        print()

        init_params(self.d, parser=parser)

    @property
    def root_dir(self):
        return self._root_dir

    @property
    def train_dir(self):
        return self._train_dir

    @property
    def train_sample_size(self):
        return self._train_sample_size

    @property
    def val_dir(self):
        return self._val_dir

    @property
    def val_sample_size(self):
        return self._val_sample_size

    @property
    def test_dir(self):
        return self._test_dir

    @property
    def test_sample_size(self):
        return self._test_sample_size

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def transfer_dir(self):
        return self._transfer_dir

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir
