"""
Setup basic hyperparameters
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import DataSet
from subsets.cub_200_2011 import read_subset
from models.resnet_v1_5 import ResNet50 as ConvNet
from optimizers import MomentumOptimizer as Optimizer
from evaluators import AccuracyEvaluator as Evaluator
from models.init_from_checkpoint import resnet_v1_50_101 as init_from_checkpoint
import utils


class Parameters(object):
    # FIXME: Directories
    _root_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # FIXME
    _train_dir = os.path.join(_root_dir, 'tfdatasets/cub-200-2011/train')
    _train_sample_size = None  # Data size. None for all files in train_dir
    _val_dir = None  # Can be None
    _val_sample_size = 500
    _test_dir = os.path.join(_root_dir, 'tfdatasets/cub-200-2011/test')
    _test_sample_size = None

    _save_dir = 'D:/trained_models/ResNet-50_CUB-200'
    _transfer_dir = None
    _checkpoint_dir = os.path.join(_root_dir, 'pretrained_models', 'resnet_v1_50', 'resnet_v1_50.ckpt')

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
    d['shuffle'] = True  # Whether to shuffle the data

    # FIXME: Transfer learning parameters
    d['init_from_public_checkpoint'] = False  # Whether to use pre-trained model in checkpoint_dir
    d['blocks_to_load'] = None  # Blocks to load variables on. None for all blocks
    d['load_logits'] = False  # Whether to load variables related to logits
    d['start_epoch'] = 0  # Start epoch to continue training from
    d['model_to_load'] = 0  # The (n+1)-th best model is loaded. Can be the name of the checkpoint file

    d['blocks_to_train'] = None  # Blocks to train. None for all blocks and [None] for logits only
    d['update_batch_norm'] = True  # Whether to update batch norm statistics. None to follow blocks_to_train

    # FIXME: Training hyperparameters
    d['argmax_output'] = False  # If True, the network's output will be argmaxed (output shape=(N, H, W, 1))
    d['cpu_offset'] = 0  # CPU device offset
    d['gpu_offset'] = 0  # GPU device offset

    d['num_gpus'] = 1
    d['batch_size'] = 8  # Total batch size (= batch_size_per_gpu*num_gpus)
    d['num_epochs'] = 300
    d['base_learning_rate'] = 0.25  # Learning rate = base_learning_rate*batch_size/256 (linear scaling rule)
    d['momentum'] = 0.9  # Momentum of optimizers

    d['learning_rate_decay_method'] = 'step'  # None, 'step', 'exponential', 'polynomial', 'cosine' (default)
    d['learning_rate_decay_params'] = (0.1, 150, 250)

    d['max_to_keep'] = 5  # Maximum number of models to save
    d['score_threshold'] = 0.0  # Model is saved if its score is better by this threshold

    # FIXME: Regularization hyperparameters
    d['l1_reg'] = 0.0  # L1 regularization factor
    d['l2_reg'] = 0.001  # L2 regularization factor

    d['label_smoothing'] = 0.0  # Label smoothing factor
    d['dropout_rate'] = 0.2  # Dropout rate

    # FIXME: Data augmentation hyperparameters
    d['augment_train'] = True  # Online augmentation for training data
    d['augment_test'] = False  # Online augmentation for validation and test data

    d['zero_pad_ratio'] = 0.0  # Zero padding ratio = (zero_padded_image_size - nn_input_size)/nn_input_size

    d['rand_affine'] = True  # Bool
    d['rand_scale'] = (1.0, 1.0)  # Minimum and maximum scaling factors (x/y)
    d['rand_ratio'] = (1.0, 1.0)  # Minimum and maximum pixel aspect ratios (x/y)
    d['rand_x_trans'] = 0.0  # Range in proportion (0.2 means +-10%)
    d['rand_y_trans'] = 0.0  # Range in proportion
    d['rand_rotation'] = 0  # Range in degrees
    d['rand_shear'] = 0  # Range in degrees
    d['rand_x_reflect'] = True  # Bool
    d['rand_y_reflect'] = False  # Bool

    d['rand_crop'] = True  # Bool
    d['rand_crop_scale'] = (0.25, 1.0)  # Scale*input_size patch crop from an image
    d['rand_crop_ratio'] = (4/5, 5/4)

    d['rand_distortion'] = True  # Bool
    d['rand_hue'] = 0.2  # Hue range in proportion
    d['rand_saturation'] = (0.8, 1.25)  # Lower and upper bounds of random saturation factors
    d['rand_contrast'] = (0.8, 1.25)  # Lower and upper bounds of random contrast factors
    d['rand_brightness'] = 0.2  # Brightness range in proportion

    d['cutmix'] = False  # CutMix augmentation

    def __init__(self):
        print('Training directory: ', self.train_dir)
        print('Test directory: ', self.test_dir)
        print('Transfer learning directory: ', self.transfer_dir)
        print('Data save directory: ', self.save_dir)
        print('')

        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = str(self.d['num_gpus'])
        os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
        os.environ['TF_SYNC_ON_FINISH'] = '0'
        # os.environ["OMP_NUM_THREADS"] = str(self.d['num_parallel_calls'])
        # os.environ["KMP_BLOCKTIME"] = '0'
        # os.environ["KMP_SETTINGS"] = '1'
        # os.environ["KMP_AFFINITY"] = 'granularity=fine,verbose,compact,1,0'

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
