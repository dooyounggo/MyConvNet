"""
Functions for data pre-processing.
Includes various image reading and resizing functions.
"""

import os
import numpy as np
import cv2

INT_TYPES = ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64']
FLOAT_TYPES = ['float16', 'float32', 'float64']


def read_subset_cls(subset_dir, shuffle=False, sample_size=None):
    filenames = os.listdir(subset_dir)
    image_dirs = []
    label_dirs = []
    for fname in filenames:
        ext = fname.split('.')[-1].lower()
        full_filename = os.path.join(subset_dir, fname)
        if ext == 'csv':
            label_dirs.append(full_filename)
        elif ext == 'jpg' or ext == 'jpeg' or ext == 'bmp':
            image_dirs.append(full_filename)

    set_size = len(image_dirs)
    if len(label_dirs) == 0:
        label_dirs = None
    else:
        assert len(image_dirs) == len(label_dirs), \
            'Number of examples mismatch: {} images vs. {} labels'.format(len(image_dirs), len(label_dirs))

    if sample_size is not None and sample_size < set_size:
        if shuffle:
            idx = np.random.choice(np.arange(set_size), size=sample_size, replace=False).astype(int)
            image_dirs = list(np.array(image_dirs)[idx])
            if label_dirs is not None:
                label_dirs = list(np.array(label_dirs)[idx])
        else:
            image_dirs = image_dirs[:sample_size]
            if label_dirs is not None:
                label_dirs = label_dirs[:sample_size]
    else:
        if shuffle:
            idx = np.arange(set_size)
            np.random.shuffle(idx)
            image_dirs = list(np.array(image_dirs)[idx])
            if label_dirs is not None:
                label_dirs = list(np.array(label_dirs)[idx])

    return image_dirs, label_dirs


def read_subset_seg(subset_dir, shuffle=False, sample_size=None):
    filenames = os.listdir(subset_dir)
    image_dirs = []
    label_dirs = []
    for fname in filenames:
        ext = fname.split('.')[-1].lower()
        full_filename = os.path.join(subset_dir, fname)
        if ext == 'png':
            label_dirs.append(full_filename)
        elif ext == 'jpg' or ext == 'jpeg' or ext == 'bmp':
            image_dirs.append(full_filename)

    set_size = len(image_dirs)
    if len(label_dirs) == 0:
        label_dirs = None
    else:
        assert len(image_dirs) == len(label_dirs), \
            'Number of examples mismatch: {} images vs. {} labels'.format(len(image_dirs), len(label_dirs))

    if sample_size is not None and sample_size < set_size:
        if shuffle:
            idx = np.random.choice(np.arange(set_size), size=sample_size, replace=False).astype(int)
            image_dirs = list(np.array(image_dirs)[idx])
            if label_dirs is not None:
                label_dirs = list(np.array(label_dirs)[idx])
        else:
            image_dirs = image_dirs[:sample_size]
            if label_dirs is not None:
                label_dirs = label_dirs[:sample_size]
    else:
        if shuffle:
            idx = np.arange(set_size)
            np.random.shuffle(idx)
            image_dirs = list(np.array(image_dirs)[idx])
            if label_dirs is not None:
                label_dirs = list(np.array(label_dirs)[idx])

    return image_dirs, label_dirs


def random_resized_crop(image, out_size, interpolation=cv2.INTER_LINEAR,
                        random=True, scale=(1.0, 1.0), ratio=(1.0, 1.0)):

    out_size_ratio = out_size[1]/out_size[0]
    in_size = image.shape
    h = in_size[0]
    w = in_size[1]

    size_h = np.sqrt(h*w/out_size_ratio)
    size_w = np.sqrt(h*w*out_size_ratio)
    image = zero_pad(image, [np.ceil(size_h).astype(int), np.ceil(size_w).astype(int)])
    if random:
        lower, upper = scale
        # a = upper**2 - lower**2
        # b = lower**2
        # randval = np.random.uniform()
        # rand_scale = np.sqrt(a*randval + b)
        rand_scale = np.random.uniform(lower, upper)

        lower, upper = ratio
        base = float(upper/lower)
        randval = np.random.uniform()
        rand_ratio = lower*np.power(base, randval)

        rand_x_scale = np.sqrt(rand_scale/rand_ratio)
        rand_y_scale = np.sqrt(rand_scale*rand_ratio)

        size_h = np.around(size_h*rand_y_scale).astype(int)
        size_w = np.around(size_w*rand_x_scale).astype(int)
    else:
        size_h = np.around(size_h).astype(int)
        size_w = np.around(size_w).astype(int)

    image = crop(image, [size_h, size_w], random=random)
    image = cv2.resize(image, dsize=tuple(out_size[1::-1]), interpolation=interpolation)

    return to_float(image)


def resize_with_crop(image, out_size, interpolation=cv2.INTER_LINEAR, random=False):
    in_size = image.shape
    h_ratio = float(in_size[0])/out_size[0]
    w_ratio = float(in_size[1])/out_size[1]

    if h_ratio < 1.0 or w_ratio < 1.0:
        resize_expand(image, out_size, interpolation=interpolation)
    else:
        image = to_float(image)

    image = crop(image, out_size, random=random)

    return image


def resize_with_pad(image, out_size, interpolation=cv2.INTER_LINEAR, random=False, pad_value=0.0):
    in_size = image.shape
    h_ratio = float(in_size[0])/out_size[0]
    w_ratio = float(in_size[1])/out_size[1]

    if h_ratio > 1.0 or w_ratio > 1.0:
        resize_fit(image, out_size, interpolation=interpolation)
    else:
        image = to_float(image)

    image = zero_pad(image, out_size, random=random, pad_value=pad_value)

    return image


def resize_fit_expand(image, out_size, interpolation=cv2.INTER_LINEAR, random=False, pad_value=0.0):
    in_size = image.shape
    h_ratio = float(in_size[0])/out_size[0]
    w_ratio = float(in_size[1])/out_size[1]

    if h_ratio > 1.0 and w_ratio > 1.0:
        image = resize_expand(image, out_size, interpolation=interpolation, random=random)
    elif h_ratio < 1.0 and w_ratio < 1.0:
        image = resize_fit(image, out_size, interpolation=interpolation, random=random)
    else:
        image = to_float(image)

    image = crop(image, out_size, random=random)
    image = zero_pad(image, out_size, random=random, pad_value=pad_value)

    return image


def resize_fit(image, out_size, interpolation=cv2.INTER_LINEAR, random=False, pad_value=0.0):
    in_size = image.shape
    h_ratio = float(in_size[0])/out_size[0]
    w_ratio = float(in_size[1])/out_size[1]

    if h_ratio > w_ratio:
        re_size = [out_size[0], int(np.round(in_size[1]/h_ratio))]
    else:
        re_size = [int(np.round(in_size[0]/w_ratio)), out_size[1]]

    image = cv2.resize(image, dsize=tuple(re_size[::-1]), interpolation=interpolation)
    image = zero_pad(image, out_size, random=random, pad_value=pad_value)

    return to_float(image)


def resize_expand(image, out_size, interpolation=cv2.INTER_LINEAR, random=False):
    in_size = image.shape
    h_ratio = float(in_size[0])/out_size[0]
    w_ratio = float(in_size[1])/out_size[1]

    if h_ratio < w_ratio:
        re_size = [out_size[0], int(np.round(in_size[1]/h_ratio))]
    else:
        re_size = [int(np.round(in_size[0]/w_ratio)), out_size[1]]

    image = cv2.resize(image, dsize=tuple(re_size[::-1]), interpolation=interpolation)
    image = crop(image, out_size, random=random)

    return to_float(image)


def resize_with_crop_or_pad(image, out_size, random=False, pad_value=0.0, *args, **kwargs):
    image = crop(image, out_size, random=random)
    image = zero_pad(image, out_size, random=random, pad_value=pad_value)

    return to_float(image)


def crop(image, out_size, random=False):
    in_size = image.shape
    out_size = list(out_size)
    h_diff = in_size[0] - out_size[0]
    w_diff = in_size[1] - out_size[1]
    assert h_diff >= 0 or w_diff >= 0, 'At least one side must be longer than or equal to the output size'

    if h_diff > 0 and w_diff > 0:
        if random:
            h_idx = np.random.randint(0, h_diff)
            w_idx = np.random.randint(0, w_diff)
        else:
            h_idx = h_diff//2
            w_idx = w_diff//2
        image = image[h_idx:h_idx + out_size[0], w_idx:w_idx + out_size[1]]
    elif h_diff > 0:
        if random:
            h_idx = np.random.randint(0, h_diff)
        else:
            h_idx = h_diff//2
        image = image[h_idx:h_idx + out_size[0], :]
    elif w_diff > 0:
        if random:
            w_idx = np.random.randint(0, w_diff)
        else:
            w_idx = w_diff//2
        image = image[:, w_idx:w_idx + out_size[1]]

    return image


def zero_pad(image, out_size, random=False, pad_value=0.0):
    in_size = image.shape
    out_size = list(out_size)
    h_diff = out_size[0] - in_size[0]
    w_diff = out_size[1] - in_size[1]
    assert h_diff >= 0 or w_diff >= 0, 'At least one side must be shorter than or equal to the output size'

    out_size_max[0] = max(out_size[0], in_size[0])
    out_size_max[1] = max(out_size[1], in_size[1])
    image_out = np.zeros(out_size_max[:2] + [in_size[-1]], dtype=image.dtype)
    if pad_value != 0.0:
        image_out += pad_value

    if h_diff > 0 and w_diff > 0:
        if random:
            h_idx = np.random.randint(0, h_diff)
            w_idx = np.random.randint(0, w_diff)
        else:
            h_idx = h_diff//2
            w_idx = w_diff//2
        image_out[h_idx:h_idx + out_size[0], w_idx:w_idx + out_size[1]] = image
    elif h_diff > 0:
        if random:
            h_idx = np.random.randint(0, h_diff)
        else:
            h_idx = h_diff//2
        image_out[h_idx:h_idx + out_size[0], :] = image
    elif w_diff > 0:
        if random:
            w_idx = np.random.randint(0, w_diff)
        else:
            w_idx = w_diff//2
        image_out[:, w_idx:w_idx + out_size[1]] = image

    return image_out


def to_float(image):
    if image.dtype in INT_TYPES:
        image = image.astype(np.float32)/255.0
    else:
        return image.astype(np.float32)
    return image


def to_int(image):
    if image.dtype in FLOAT_TYPES:
        image = (image*255).astype(np.int32)
    else:
        image = image.astype(np.int32)
    return image
