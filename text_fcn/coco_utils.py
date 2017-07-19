# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from six.moves import range
from six.moves import urllib

import os
import sys
import tarfile
import zipfile

import cv2
import numpy as np


URL = 'https://s3.amazonaws.com/cocotext/COCO_Text.zip'


def coco_read_dataset(ct):
    """
    Returns train and validation dataset filenames list (no extension)
    :param ct: COCO_Text instance
    """
    train = [
        i for i in ct.train if any(
            ann for ann in ct.imgToAnns[i]
            if ct.anns[ann]['legibility'] == 'legible'
        )
    ]
    val = [
        i for i in ct.val if any(
            ann for ann in ct.imgToAnns[i]
            if ct.anns[ann]['legibility'] == 'legible'
        )
    ]
    val = ct.val
    test = [ct.imgs[i]['file_name'][:-4] for i in ct.test]

    return train, val, test


def synth_read_dataset(st):
    """
    Returns train and validation dataset filenames list (no extension)
    :param st: Synth_Text .npy
    """
    train = [
        key for key in st
        if st[key]['set'] == 'train'
    ]
    val = [
        key for key in st
        if st[key]['set'] == 'val'
    ]
    return train, val, None


def icdar_read_dataset(dt):
    """
    Returns train and validation dataset filenames list (no extension)
    :param st: ICDAR2015 .npy
    """
    train = [
        key for key in dt
        if dt[key]['set'] == 'train'
    ]
    val = [
        key for key in dt
        if dt[key]['set'] == 'val'
    ]
    test = [
        key for key in dt
        if dt[key]['set'] == 'test'
    ]

    return train, val, test

def maybe_download_and_extract(dir_path, url_name, is_tarfile=False, is_zipfile=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if is_tarfile:
            tarfile.open(filepath, 'r:gz').extractall(dir_path)
        elif is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)


def get_window(shape, bbox):
    """
    :param shape: image shape (used as boundaries)
    :param bbox: initial bbox from which to expand (x, y, w, h)
    :return: window dict with slices and padding values
    """
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
    h, w = shape

    # Bounds checking is needed thanks to SynthText :)
    x1 = max(0, x1)
    x2 = min(w, x2)
    y1 = max(0, y1)
    y2 = min(h, y2)

    # expand left, right, up, down
    x1 -= np.random.randint(0, x1 + 1)
    x2 += np.random.randint(0, w - x2 + 1)
    y1 -= np.random.randint(0, y1 + 1)
    y2 += np.random.randint(0, h - y2 + 1)

    ratio = (x2 - x1) / (y2 - y1)

    try:
        if ratio > 1:
            # expand ys (window's width)
            diff = (x2 - x1) - (y2 - y1)
            split = np.random.randint(0, diff)
            y1 -= split
            y2 += diff-split
        elif ratio < 1:
            # expand xs (window's height)
            diff = (y2 - y1) - (x2 - x1)
            split = np.random.randint(0, diff)
            x1 -= split
            x2 += diff-split
        else:
            # already a square
            pass
    except:
        print(ratio)
        print(x1, x2, y1, y2)

    assert((x2 - x1) == (y2 - y1))

    return {
        'slice': [
            slice(max(0, y1), y2),
            slice(max(0, x1), x2)
        ],
        'pad': (
            (max(0, -y1), max(0, y2-h)),
            (max(0, -x1), max(0, x2-w))
        )
    }


def crop_resize(images, window, size):
    """
    :param images: list of (image, annotation, weight)
    :param window: window returned by get_window()
    :param size: resize size (square)
    :return: cropped images
    """
    pad = [((0, 0),), (), ()]
    interp = [cv2.INTER_CUBIC, cv2.INTER_NEAREST, cv2.INTER_NEAREST]
    dsize = (size, size)
    value = [0.0, 0.0, 1.0]

    images[0] = images[0][window['slice']]
    images[0] = np.pad(images[0],
                        window['pad'] + pad[0],
                        'constant',
                        constant_values=value[0])
    images[0] = cv2.resize(images[0], dsize, interpolation=interp[0])

    ann_fst = images[1][:,:,0][window['slice']]
    ann_fst = np.pad(ann_fst,
                     window['pad'] + pad[1],
                     'constant',
                     constant_values=value[1])
    ann_fst = cv2.resize(ann_fst, dsize, interpolation=interp[1])

    ann_snd = images[1][:,:,1][window['slice']]
    ann_snd = np.pad(ann_snd,
                     window['pad'] + pad[1],
                     'constant',
                     constant_values=value[1])
    ann_snd = cv2.resize(ann_snd, dsize, interpolation=interp[1])

    images[1] = np.dstack((ann_fst, ann_snd))

    images[2] = images[2][window['slice']]
    images[2] = np.pad(images[2],
                       window['pad'] + pad[2],
                       'constant',
                       constant_values=value[2])
    images[2] = cv2.resize(images[2], dsize, interpolation=interp[2])

    return images
