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


def read_dataset(ct):
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
    test = [ct.imgs[i]['file_name'][:-4] for i in ct.test]

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


def get_window(shape, annotation):
    """
    :param shape: image shape (used as boundaries)
    :param annotation:  ct.anns object
    :return: window dict with slices and padding values
    """
    x, y, dx, dy = np.array(annotation['bbox'], np.int32)
    x1, y1, x2, y2 = x, y, x+dx, y+dy
    h, w = shape

    # expand left, right, up, down
    x1 -= np.random.randint(0, x1+1)
    x2 += np.random.randint(0, w-x2+1)
    y1 -= np.random.randint(0, y1+1)
    y2 += np.random.randint(0, h-y2+1)

    ratio = (x2 - x1) / (y2 - y1)

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

    for i in range(3):
        images[i] = images[i][window['slice']]
        images[i] = np.pad(images[i],
                           window['pad'] + pad[i],
                           'constant',
                           constant_values=value[i])
        images[i] = cv2.resize(images[i], dsize, interpolation=interp[i])

    return images
