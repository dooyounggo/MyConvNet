"""
Setup various hyperparameters
"""

import os
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from initialization import init_params
from dataset import DataSet
from subsets.places365_standard import read_subset
from models.upinet import UPINet as ConvNet
from optimizers import AdamOptimizer as Optimizer
from evaluators import PSNREvaluator as Evaluator
from models.init_from_checkpoint import resnet_v1_50_101 as init_from_checkpoint
import utils


class Parameters(object):
    # FIXME: Directories
    _root_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # FIXME
    _train_dir = os.path.join(_root_dir, 'tfdatasets/places365/test')
    _train_sample_size = None  # Data size. None for all files in train_dir
    _val_dir = os.path.join(_root_dir, 'tfdatasets/places365/validation')  # Can be None
    _val_sample_size = 5000
    _test_dir = os.path.join(_root_dir, 'tfdatasets/places365/validation')
    _test_sample_size = 5000

    _save_dir = os.path.join(_root_dir, 'trained_models/UPINet_Places365')
    _transfer_dir = None
    _checkpoint_dir = os.path.join(_root_dir, 'pretrained_models', 'resnet_v1_50', 'resnet_v1_50.ckpt')

    d = dict()
    # FIXME: Image pre-processing hyperparameters
    d['image_size'] = (128, 128, 3)  # Processed image size
    d['image_size_test'] = None  # If None, same as 'image_size'
    d['resize_type'] = 'resize_with_crop_or_pad'  # Resize types: 'resize', 'resize_expand', 'random_resized_crop', ...
    d['resize_type_test'] = None  # If None, same as 'resize_type'
    d['resize_random'] = True  # Randomness of padding and crop operations
    d['resize_random_test'] = False

    d['input_size'] = (128, 128, 3)  # Network input size after augmentation
    d['image_mean'] = 0.5  # If None, it will be calculated and it may take some time
    d['zero_center'] = False
    d['shuffle'] = True  # Whether to shuffle the data

    # FIXME: Transfer learning parameters
    d['init_from_public_checkpoint'] = False  # Whether to use pre-trained model in checkpoint_dir

    d['start_epoch'] = 0  # Start epoch to continue training from
    d['model_to_load'] = 0  # The (n+1)-th best model is loaded. Can be the name of the checkpoint file

    d['blocks_to_train'] = None  # List of blocks to train. None for all blocks and [None] for logits only
    d['update_batch_norm'] = True  # Whether to update batch norm statistics. None to follow blocks_to_train

    # FIXME: Training hyperparameters
    d['half_precision'] = False  # If True, the float16 data type is used
    d['channel_first'] = False  # If True, the "NCHW" format is used instead of "NHWC"
    d['cpu_offset'] = 0  # CPU device offset
    d['gpu_offset'] = 0  # GPU device offset

    d['num_gpus'] = 1
    d['batch_size'] = 16  # Total batch size (= batch_size_per_gpu*num_gpus)
    d['num_epochs'] = 150
    d['base_learning_rate'] = 0.0016  # Learning rate = base_learning_rate*batch_size/256 (linear scaling rule)
    d['momentum'] = 0.9  # Momentum of optimizers
    d['moving_average_decay'] = 0.9999  # Decay rate of exponential moving average
    d['batch_norm_decay'] = 0.999  # Decay rate of batch statistics

    d['learning_rate_decay_method'] = None  # None, 'step', 'exponential', 'polynomial', 'cosine' (default)
    d['learning_rate_decay_params'] = 0.0
    d['learning_warmup_epoch'] = 1.0  # Linear warmup epoch

    d['max_to_keep'] = 5  # Maximum number of models to save
    d['score_threshold'] = 0.0  # Model is saved if its score is better by this threshold

    # FIXME: Regularization hyperparameters
    d['denoising_loss_factor'] = 0.0
    d['edge_loss_l1_factor'] = 0.0
    d['edge_loss_l2_factor'] = 0.0
    d['edge_loss_true_ratio'] = 0.0
    d['l1_reg'] = 0.0  # L1 regularization factor
    d['l2_reg'] = 0.0  # L2 regularization factor

    # FIXME: Data augmentation hyperparameters
    d['augment_train'] = True  # Online augmentation for training data
    d['augment_test'] = False  # Online augmentation for validation and test data

    d['zero_pad_ratio'] = 0.0  # Zero padding ratio = (zero_padded_image_size - nn_input_size)/nn_input_size

    d['rand_blur_stddev'] = 0.5  # Maximum sigma for Gaussian blur

    d['rand_affine'] = True  # Bool
    d['rand_scale'] = (1.0, 1.0)  # Minimum and maximum scaling factors (x/y)
    d['rand_ratio'] = (1.0, 1.0)  # Minimum and maximum pixel aspect ratios (x/y)
    d['rand_x_trans'] = 0.0  # Range in proportion (0.2 means +-10%)
    d['rand_y_trans'] = 0.0  # Range in proportion
    d['rand_rotation'] = 0  # Range in degrees
    d['rand_shear'] = 0  # Range in degrees
    d['rand_x_reflect'] = True  # Bool
    d['rand_y_reflect'] = False  # Bool

    d['rand_crop'] = False  # Bool
    d['rand_crop_scale'] = (0.25, 1.75)  # Scale*input_size patch is cropped from an image
    d['rand_crop_ratio'] = (3/4, 4/3)

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
