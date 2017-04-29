#coding=utf-8

import os
import sys
import math
import random
import tarfile
import zipfile
import numpy as np
import scipy.misc as misc
from six.moves import urllib



URL = 'https://s3.amazonaws.com/cocotext/COCO_Text.zip'


def read_dataset(ct):
    """
    Returns train and validation dataset filenames list (no extension)
    :param ct: COCO_Text instance
    """
    # legible annotations
    valid_anns = [
        ann for ann in ct.anns
    ]

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
    test = [i for i in ct.test]

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

def to_mask(img):
    """
    :param img: B/W image, values in {0, 255}
    :return:    image in {0, 1}
    """
    img[img > 0] = 1
    return img.astype(np.int32)

def to_ann(res):
    """
    :param res: resulting image from to_mask, values in {0, 1}
    :return:    image in {0, 255}
    """
    res[res > 0] = 255
    return res.astype(np.uint8)
