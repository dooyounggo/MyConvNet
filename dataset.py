"""
Manage datasets using the TensorFlow Data API.
"""

import numpy as np
import tensorflow.compat.v1 as tf
import csv
import cv2
import subsets.subset_functions as sf


class DataSet(object):
    IMAGE_ONLY = None
    IMAGE_CLASSIFICATION = 'image_classification'
    IMAGE_SEGMENTATION = 'image_segmentation'
    SEMANTIC_SEGMENTATION = IMAGE_SEGMENTATION
    DCGAN = 'deep_convolutional_gan'
    IMAGE_TO_IMAGE_TRANSLATION = 'image_to_image_translation'
    # OBJECT_DETECTION = 'object_detection'
    # INSTANCE_SEGMENTAION = 'instance_segmentation'
    # MULTI_LABEL_IMAGE_CLASSIFICATION = 'multi-label_image_classification'

    RANDOM_RESIZED_CROP = 'random_resized_crop'
    RESIZE = 'resize'
    RESIZE_FIT = 'resize_fit'
    RESIZE_EXPAND = 'resize_expand'
    RESIZE_FIT_EXPAND = 'resize_fit_expand'
    RESIZE_WITH_CROP_OR_PAD = 'resize_with_crop_or_pad'
    PADDED_RESIZE = 'padded_resize'

    INTERPOLATION_NEAREST = 'nearest'
    INTERPOLATION_BILINEAR = 'bilinear'
    INTERPOLATION_BICUBIC = 'bicubic'

    def __init__(self, image_dirs, label_dirs=None, task_type=IMAGE_ONLY, class_names=None, num_classes=None,
                 out_size=None, resize_method=None, resize_randomness=False, shuffle_data=None, from_memory=False,
                 **kwargs):
        """
        :param image_dirs: list or tuple, paths to images
        :param label_dirs: list or tuple, paths to labels. If None, fake labels are created.
        :param task_type: string, type of the task for which the dataset is intended.
        :param class_names: list or tuple, names of each class. Used to count the number of classes.
        :param out_size: list or tuple, size of images to be used for training.
        :param resize_method: string, resizing method for image preprocessing.
        :param resize_randomness: Bool, randomness of resize operations such as crop and padding.
        :param from_memory: Bool, if true, image_dirs and label_dirs must be numpy arrays.
        :param kwargs: dict, extra arguments containing hyperparameters.
        """
        if image_dirs is None:
            image_dirs = [np.nan for _ in label_dirs]  # Fake images
        if label_dirs is None:
            label_dirs = [np.nan for _ in image_dirs]  # Fake labels
        assert len(image_dirs) == len(label_dirs), 'Number of examples mismatch, between images and labels'

        self._image_dirs = image_dirs
        self._label_dirs = label_dirs
        if out_size is None:
            self._image_size = kwargs.get('image_size', (256, 256, 3))
        else:
            self._image_size = out_size

        self._task_type = task_type

        if resize_method is None:
            self._resize_method = kwargs.get('resize_type', 'resize')
        else:
            self._resize_method = resize_method
        self._resize_randomness = resize_randomness
        self._resize_interpolation = kwargs.get('resize_interpolation', 'bilinear')

        if shuffle_data is None:
            self._shuffle = kwargs.get('shuffle', True)
        else:
            self._shuffle = shuffle_data

        self._from_memory = from_memory

        if class_names is None:
            if self.task_type in [DataSet.IMAGE_CLASSIFICATION, DataSet.IMAGE_SEGMENTATION, DataSet.DCGAN]:
                assert num_classes is not None, 'Either class_names or num_classes must be provided.'
            self._num_classes = num_classes
        else:
            self._num_classes = len(class_names)
        self._class_names = class_names

        self._cpu_offset = kwargs.get('cpu_offset', 0)
        self._gpu_offset = kwargs.get('gpu_offset', 0)
        num_gpus = kwargs.get('num_gpus', None)
        if num_gpus is None:
            num_gpus = 1 if tf.test.is_gpu_available(cuda_only=True) else 0
        if num_gpus == 0:  # No GPU available
            self._num_shards = 1
            self._compute_device = 'cpu'
            dev_offset = self.cpu_offset
        else:
            self._num_shards = num_gpus
            self._compute_device = 'gpu'
            dev_offset = 0
        self._device_offset = dev_offset

        self._batch_size = kwargs.get('batch_size', 16)

        self._parameters = kwargs

        self._image_mean = kwargs.get('image_mean', 0.5)
        self._num_examples = len(image_dirs)
        self._examples_per_class = None
        self._balanced_weights = None

        self._datasets = []
        self._iterators = []
        self._handles = []
        batch_size_per_gpu = self.batch_size//self.num_shards

        idx = np.arange(self.num_examples)
        num_split = self.num_examples//self.batch_size
        exact_size = self.batch_size*num_split
        splitted = list(np.split(idx[:exact_size], num_split))
        idx_shards = []
        for sp in splitted:
            idx_shards += list(np.split(sp, self.num_shards))
        idx_shards += np.array_split(idx[exact_size:], self.num_shards)

        image_dirs = np.array(image_dirs)
        label_dirs = np.array(label_dirs)
        with tf.name_scope('dataset/'):
            with tf.device('/cpu:{}'.format(self.cpu_offset)):
                for i in range(self.num_shards):
                    idx = np.concatenate(idx_shards[i::self.num_shards])
                    dataset = tf.data.Dataset.from_tensor_slices((image_dirs[idx], label_dirs[idx]))
                    if self.shuffle:
                        dataset = dataset.shuffle(buffer_size=min([np.ceil(self.num_examples/self.num_shards),
                                                                   np.ceil(1024*batch_size_per_gpu/32)]))
                    if not self.from_memory:
                        dataset = dataset.map(lambda image_dir, label_dir: tuple(tf.py_func(self._load_function,
                                                                                            (image_dir, label_dir),
                                                                                            (tf.float32, tf.float32))),
                                              num_parallel_calls=kwargs.get('num_parallel_calls', 4)//self.num_shards)
                    dataset = dataset.batch(batch_size_per_gpu)
                    dataset = dataset.apply(tf.data.experimental.copy_to_device('/{}:{}'.format(self.compute_device,
                                                                                                i + dev_offset)))
                    with tf.device('/{}:{}'.format(self.compute_device, i + dev_offset)):
                        dataset = dataset.prefetch(buffer_size=1)
                        self._datasets.append(dataset)
                        iterator = dataset.make_initializable_iterator()
                        self._iterators.append(iterator)
                        self._handles.append(iterator.string_handle())
        self._init_count = 0

    @property
    def image_dirs(self):
        return self._image_dirs

    @property
    def label_dirs(self):
        return self._label_dirs

    @property
    def image_size(self):
        return self._image_size

    @property
    def task_type(self):
        return self._task_type

    @property
    def num_shards(self):
        return self._num_shards

    @property
    def cpu_offset(self):
        return self._cpu_offset

    @property
    def gpu_offset(self):
        return self._gpu_offset

    @property
    def compute_device(self):
        return self._compute_device

    @property
    def device_offset(self):
        return self._device_offset

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def shuffle(self):
        return self._shuffle

    @property
    def resize_method(self):
        return self._resize_method

    @property
    def resize_randomness(self):
        return self._resize_randomness

    @property
    def resize_interpolation(self):
        return self._resize_interpolation

    @property
    def from_memory(self):
        return self._from_memory

    @property
    def image_mean(self):
        return self._image_mean

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def class_names(self):
        return self._class_names

    @property
    def examples_per_class(self):
        return self._examples_per_class

    @property
    def balanced_weights(self):
        return self._balanced_weights

    @property
    def datasets(self):
        return self._datasets

    @property
    def iterators(self):
        return self._iterators

    @property
    def handles(self):
        return self._handles

    @property
    def init_count(self):
        return self._init_count

    def _load_function(self, image_dir, label_dir):
        if isinstance(image_dir, bytes):
            image_dir = image_dir.decode()
        if isinstance(label_dir, bytes):
            label_dir = label_dir.decode()

        if not isinstance(image_dir, str):  # No image. A fake image is generated.
            image = np.zeros(self.image_size, dtype=np.float32)
        else:
            image = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)
            interpolation_method = self.resize_interpolation.lower()
            if interpolation_method == 'nearest' or interpolation_method == 'nearest neighbor':
                interpolation = cv2.INTER_NEAREST
            elif interpolation_method == 'bilinear':
                interpolation = cv2.INTER_LINEAR
            elif interpolation_method == 'bicubic':
                interpolation = cv2.INTER_CUBIC
            else:
                raise ValueError('Interpolation method \"{}\" is not supported.'.format(self.resize_interpolation))
            image = self._resize_function(image, self.image_size, interpolation=interpolation)

        if not isinstance(label_dir, str):  # No label
            label = np.array(np.nan, dtype=np.float32)
        else:   # Note that the labels are not one-hot encoded.
            if self.task_type == DataSet.IMAGE_ONLY:
                label = np.array(np.nan, dtype=np.float32)

            elif self.task_type == DataSet.IMAGE_CLASSIFICATION:
                ext = label_dir.split('.')[-1].lower()
                if ext == 'csv':
                    with open(label_dir, 'r', encoding='utf-8') as f:
                        rdr = csv.reader(f)
                        line = next(rdr)
                    label = int(line[0])
                elif ext == 'txt':
                    with open(label_dir, 'r', encoding='utf-8') as f:
                        line = f.readline()
                    label = int(line.rstrip())
                else:
                    raise ValueError('Label file extension \".{}\" is not supported for {}'.format(ext, self.task_type))
                label = np.array(label, dtype=np.float32)

            elif self.task_type == DataSet.IMAGE_SEGMENTATION:
                ext = label_dir.split('.')[-1].lower()
                if ext in sf.IMAGE_FORMATS:
                    label = cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE)
                    label = self._resize_function(label, self.image_size, interpolation=cv2.INTER_NEAREST)
                    label = np.round(label[..., 0]*255)
                else:
                    raise ValueError('Label file extension \".{}\" is not supported for {}'.format(ext, self.task_type))

            elif self.task_type == DataSet.IMAGE_TO_IMAGE_TRANSLATION:
                ext = label_dir.split('.')[-1].lower()
                if ext in sf.IMAGE_FORMATS:
                    label = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)
                    interpolation_method = self.resize_interpolation.lower()
                    if interpolation_method == 'nearest' or interpolation_method == 'nearest neighbor':
                        interpolation = cv2.INTER_NEAREST
                    elif interpolation_method == 'bilinear':
                        interpolation = cv2.INTER_LINEAR
                    elif interpolation_method == 'bicubic':
                        interpolation = cv2.INTER_CUBIC
                    else:
                        raise ValueError('Interpolation method \"{}\" is not supported.'
                                         .format(self.resize_interpolation))
                    label = self._resize_function(label, self.image_size, interpolation=interpolation)
                else:
                    raise ValueError('Label file extension \".{}\" is not supported for {}'.format(ext, self.task_type))

            else:
                raise ValueError('\"{}\" task is not supported'.format(self.task_type))

        return image, label

    def _resize_function(self, image, image_size, **kwargs):
        interpolation = kwargs.get('interpolation', cv2.INTER_LINEAR)
        pad_value = kwargs.get('pad_value', 0.0)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :-1]
        if image.shape[-1] == 1:
            image = np.tile(image, (1, 1, 3))

        resize_method = self.resize_method.lower()
        if resize_method == 'resize':
            image = sf.to_float(cv2.resize(image, dsize=tuple(image_size[1::-1]), interpolation=interpolation))
        elif resize_method == 'resize_fit':
            image = sf.resize_fit(image, image_size, interpolation=interpolation, random=self.resize_randomness,
                                  pad_value=pad_value)
        elif resize_method == 'resize_expand':
            image = sf.resize_expand(image, image_size, interpolation=interpolation, random=self.resize_randomness)
        elif resize_method == 'resize_fit_expand':
            image = sf.resize_fit_expand(image, image_size, interpolation=interpolation, random=self.resize_randomness,
                                         pad_value=pad_value)
        elif resize_method == 'resize_with_crop_or_pad':
            image = sf.resize_with_crop_or_pad(image, image_size, random=self.resize_randomness, pad_value=pad_value)
        elif resize_method == 'padded_resize' or resize_method == 'pad_resize':
            scale = self._parameters.get('padded_resize_scale', 2.0)
            image = sf.padded_resize(image, image_size, interpolation=interpolation,
                                     random=self.resize_randomness, scale=scale, pad_value=pad_value)
        elif resize_method == 'random_resized_crop' or resize_method == 'random_resize_crop':
            prefixes = ['rand_resized_crop', 'random_resized_crop', 'rand_resize_crop', 'random_resize_crop',
                        'resized_crop', 'resize_crop', 'rand_resize', 'random_resize']
            scale, ratio, padding = None, None, True
            for prefix in prefixes:
                if prefix + '_scale' in self._parameters:
                    scale = self._parameters[prefix + '_scale']
                if prefix + '_ratio' in self._parameters:
                    ratio = self._parameters[prefix + '_ratio']
                if prefix + '_padding' in self._parameters:
                    padding = self._parameters[prefix + '_padding']
            if scale is None:
                scale = (0.08, 1.0)
            if scale[0] > scale[1]:
                scale = scale[::-1]
            if ratio is None:
                ratio = (3/4, 4/3)
            if ratio[0] > ratio[1]:
                ratio = ratio[::-1]
            max_attempts = self._parameters.get('max_crop_attempts', 10)
            min_object_size = self._parameters.get('min_object_size', None)
            image = sf.random_resized_crop(image, image_size, interpolation=interpolation,
                                           random=self.resize_randomness, scale=scale, ratio=ratio,
                                           max_attempts=max_attempts, min_object_size=min_object_size,
                                           padding=padding, pad_value=pad_value)
        else:
            raise ValueError('Resize type of {} is not supported.'.format(self.resize_method))

        image = image[..., 0:self.image_size[-1]]
        return image

    def data_statistics(self, verbose=True, max_data=10000):
        image_mean = 0.0
        examples = np.zeros(self.num_classes, dtype=np.int64)
        image_dirs = self.image_dirs[:max_data]
        label_dirs = self.label_dirs[:max_data]
        for n, (idir, ldir) in enumerate(zip(image_dirs, label_dirs)):
            if verbose and n % 1000 == 0:
                print('Calculating data statistics: {:5d}/{}...'.format(n, len(image_dirs)))

            image, label = self._load_function(idir, ldir)

            image_mean += np.mean(image, axis=(0, 1))/self.num_examples

            if len(label.shape) == 0:
                examples[int(label)] += 1
            else:
                for i in range(0, self.num_classes):
                    c = np.equal(label, i + 1).sum()
                    examples[i] += c

        self._image_mean = image_mean
        self._examples_per_class = examples
        self._balanced_weights = self._calculate_balanced_weights()

        if verbose:
            print('Image mean:', image_mean)
            print('Number of examples per class:')
            total_examples = self.examples_per_class.sum()
            for i in range(self.num_classes):
                print('{}: {:-5,} ({:.2%})\t'.format(self.class_names[i], examples[i], examples[i]/total_examples),
                      end='')
                if (i + 1) % 5 == 0:
                    print('')
            print('')

    def _calculate_balanced_weights(self):
        w = self.examples_per_class.sum()/self.num_classes/self.examples_per_class
        w[np.where(self.examples_per_class == 0)] = 1.0

        # FIXME: Re-balancing based on effective gradient magnitudes (experimental)
        w = np.sqrt(w)
        alpha = self.num_classes/(1/w).sum()
        w = w/alpha

        return w

    def initialize(self, sess):
        initializers = []
        for it in self.iterators:
            initializers.append(it.initializer)
        sess.run(initializers)
        self._init_count += 1

    def get_string_handles(self, sess):
        return sess.run(self.handles)
