import os
import tensorflow as tf
import subsets.oxford_iiit_two_class_seg as subset
from models.gcn import GCN as ConvNet
from evaluators import MeanIoUEvaluator as Evaluator
from models.init_from_checkpoint import resnet_v2_50_101 as init_from_pretrained_model


class Parameters(object):
    """
    Parameter initialization
    """
    # FIXME: Image parameters
    _root_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))     # Parent directory  # FIXME
    _train_dir = os.path.join(_root_dir, 'tfdatasets/oxford-iiit-pet-dataset-sualab/train')
    _train_sample_size = None   # Data size. None for all files in train_dir
    _val_dir = None  # Can be None
    _val_sample_size = None
    _test_dir = os.path.join(_root_dir, 'tfdatasets/oxford-iiit-pet-dataset-sualab/test')
    _test_sample_size = None
    _save_dir = 'D:/trained_models/gcn_iiitcatdog-384'

    _transfer_dir = None
    _pretrained_dir = os.path.join(_root_dir, 'pretrained_models', 'resnet_v2_50_2017_04_14', 'resnet_v2_50.ckpt')

    d = dict()
    d['image_size'] = (461, 461, 3)
    d['input_size'] = (384, 384, 3)
    d['image_mean'] = 0.5       # To be calculated
    d['zero_center'] = True     # Whether to zero-center the images
    d['shuffle'] = True         # Whether to shuffle the data
    d['resize_type'] = 'resize_fit'     # Resize types: 'resize', 'resize_fit', 'resize_expand', 'resize_fit_expand'
    d['num_parallel_calls'] = 4  # Number of parallel operations for dataset.map function

    d['max_to_keep'] = 5  # Maximum number of models to save
    d['score_threshold'] = 0.0  # A model is saved if its score is better by this threshold
    d['model_to_load'] = 0  # The (n+1)-th best model is loaded for the test. None for the latest

    # FIXME: Transfer learning parameters
    d['init_from_pretrained_model'] = False  # Whether to use pre-trained model in _pretrained_dir
    d['blocks_to_load'] = None  # Blocks to load variables on. None for all blocks
    d['load_logits'] = True  # Whether to load variables related to logits
    d['load_moving_average'] = True  # Whether to load exponential moving averages of variables onto variables
    d['start_epoch'] = 0  # Start epoch to continue training from

    d['blocks_to_train'] = None  # Blocks to train. None for all blocks and [None] for logits only
    d['train_batch_norm'] = True  # Whether to train batch normalization variables. None to follow blocks_to_train

    # FIXME: Training hyperparameters
    d['data_type'] = tf.float32  # Try tf.float16 if your GPU supports half-precision
    d['channel_first'] = True  # If true, NCHW format is used instead of NHWC
    d['num_gpus'] = 1
    d['batch_size'] = 4
    d['num_epochs'] = 75
    d['validation_frequency'] = None  # Validate every x iterations. None for every epoch
    d['summary_frequency'] = None  # Tensorboard summary every x iterations. None for every epoch

    d['init_learning_rate'] = 0.0025
    d['momentum'] = 0.9
    d['moving_average_decay'] = 0.9994
    d['batch_norm_decay'] = 0.9994
    d['gradient_threshold'] = 5.0
    d['loss_weighting'] = 'balanced'    # None, 'balanced', [class0_weight, class1_weight, ...]
    d['loss_scaling_factor'] = 128

    d['learning_rate_decay_method'] = 'cosine'  # 'step', 'exponential', 'cosine' (default)
    d['learning_rate_decay_params'] = (0.94, 2)
    d['learning_warmup_epoch'] = 5.0  # Linear warmup epoch

    # FIXME: Data augmentation parameters
    d['augment_factor'] = None  # int. Offline augmentation factor. None for no augmentation
    d['augment_train'] = True   # Online augmentation for training data
    d['augment_pred'] = False   # Online augmentation for validation or test data

    d['zero_pad_ratio'] = 0.1   # Zero padding ratio = (zero_padded_image_size - nn_input_size) / nn_input_size

    d['rand_scale'] = (1.0, 1.0)  # Minimum and maximum scaling factors (x/y)
    d['rand_ratio'] = (1.0, 1.0)  # Minimum and maximum pixel aspect ratios (x/y)
    d['rand_x_trans'] = 0.0  # Range in proportion (0.2 means +-10%)
    d['rand_y_trans'] = 0.0  # Range in proportion
    d['rand_rotation'] = 20  # Range in degrees
    d['rand_shear'] = 0  # Range in degrees
    d['rand_x_reflect'] = True  # Bool
    d['rand_y_reflect'] = False  # Bool
    d['rand_crop'] = True  # Bool
    d['rand_crop_scale'] = (0.3**2, 1.2**2)  # Scale*input_size patch crop from an image
    d['rand_crop_ratio'] = (0.7, 1.43)

    d['rand_hue'] = 0.2  # Hue range in proportion
    d['rand_saturation'] = (0.625, 1.6)  # Lower and upper bounds of random saturation factors
    d['rand_color_balance'] = (1.0, 1.0)  # Each color channel is multiplied by a random value
    d['rand_equalization'] = 0.3  # Equalization probability
    d['rand_contrast'] = (0.625, 1.6)  # Lower and upper bounds of random contrast factors
    d['rand_brightness'] = 0.3  # Brightness range in proportion
    d['rand_noise_mean'] = 0.0
    d['rand_noise_stddev'] = 0.0
    d['rand_solarization'] = (0.0, 0.95)  # Solarization thresholds
    d['rand_posterization'] = (5.0, 8.0)  # Posterization bits

    d['cutmix'] = False  # CutMix augmentation

    # FIXME: Regularization hyperparameters
    d['l1_reg'] = 0.0  # L1 regularization factor
    d['l2_reg'] = 0.0001  # L2 regularization factor
    d['dropout_rate'] = 0.0  # Dropout rate
    d['dropout_weights'] = False
    d['dropout_logits'] = False
    d['initial_drop_rate'] = 0.0  # Initial drop rate for stochastic depth
    d['final_drop_rate'] = 0.0  # Final drop rate for stochastic depth

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
    def pretrained_dir(self):
        return self._pretrained_dir
