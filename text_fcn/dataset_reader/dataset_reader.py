# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np


def shuffle_bgr(img):
    return np.random.permutation(img.transpose(2,0,1)).transpose(1,2,0)


def rotate(img, angle):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))


class BatchDataset(object):

    def __init__(self,
                 names,
                 batch_size,
                 image_size,
                 image_op=None,
                 augment_data=True):
        """
        Intialize a generic file reader with batching for list of files
        :param names: list of ids/names/files for the dataset reader
        :param batch_size: size of batch
        :param image_size:
        :param image_op:
        """
        print('Initializing Batch Dataset Reader...')
        self.names = np.array(names)
        np.random.shuffle(self.names)
        self.size = self.names.shape[0]
        self.batch_offset = 0
        self.epoch = 1
        self.batch_size = batch_size
        self.image_size = image_size
        # image_op passed to constructor or identity function
        self.image_op = image_op or (lambda *args, **kwargs: args)
        self.augment_data = augment_data

        if batch_size == 1:
            self._read_batch = self._simple_read
        else:
            self._read_batch = self._batch_read

        print('Image size: %d\nBatch size: %d' % (self.image_size, self.batch_size))

    def next_batch(self):
        if (self.batch_offset + self.batch_size) > self.size:
            # Epoch finished, shuffle filenames
            self.epoch += 1
            np.random.shuffle(self.names)
            # Start next epoch
            self.batch_offset = 0

        batch = slice(self.batch_offset, self.batch_offset + self.batch_size)
        self.batch_offset += self.batch_size

        return self._read_batch(batch)

    def _read_batch(self, pos):
        """
        Placeholder method to be ASSIGNED to either _simple_read or _batch_read
        """
        pass

    def _get_image(self, names):
        """
        Placeholder method to be DEFINED by specific dataset subclasses
        """
        pass

    def _batch_read(self, pos):
        """
        Read multiple images/annotations/weights and crop to same size.
        This should be used when random crops are to be used.
        :param pos: slice object
        """

        n = pos.stop - pos.start
        size = self.image_size

        images = np.zeros((n, size, size, 3), dtype=np.uint8)
        annotations = np.zeros((n, size, size, 1), dtype=np.uint8)
        weights = np.zeros((n, size, size, 1), dtype=np.float32)
        names = np.zeros(n, dtype=object)

        for i, name in enumerate(self.names[pos]):
            image, annotation, weight = self.image_op(*self._get_image(name), name=name)
            if self.augment_data:
                angle = np.random.randint(-45, 46)
                image = rotate(image, angle)
                annotation = rotate(annotation, angle)
            images[i] = image
            annotations[i] = annotation[:, :, None]
            weights[i] = weight[:, :, None]
            names[i] = name

        return images, annotations, weights, names

    def _simple_read(self, pos):
        """
        Read only 1 image/annotation/weight of any size.
        This should be used only when batch size is equal to 1 and no randomness
        on the dataset is allowed.
        :param pos: slice object of length 1
        """
        assert ((pos.stop - pos.start) == 1 == self.batch_size)
        name = self.names[pos][0]
        image, annotation, weight = self.image_op(*self._get_image(name), name=name)
        if self.augment_data:
            angle = np.random.randint(-45, 46)
            image = rotate(image, angle)
            annotation = rotate(annotation, angle)
        # Add batch_dim + convert to floating point in [0,1]
        image = np.expand_dims(image, axis=0)
        # [None,:,:,None] ==> expand_dims(_, axis=[0,3]) // syntax not supported
        annotation = annotation[None, :, :, None]
        weight = weight[None, :, :, None]
        name = np.array([name])

        return image, annotation, weight, name
