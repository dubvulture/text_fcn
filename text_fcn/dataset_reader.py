#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import os

import cv2
import numpy as np

from text_fcn import coco_utils



# Default options for no cropping (therefore batch size = 1)
NO_BATCH_OPT = {
    'batch': 1,
    'size': None
}


class BatchDataset:

    def __init__(self, coco_ids, coco_dir, ct, image_options=None, pre_saved=False):
        """
        Intialize a generic file reader with batching for list of files
        :param coco_ids: list of images' coco ids
        :param coco_dir: directory of COCO Dataset
        :param ct: COCO_Text instance
        :param image_options: dictionary of options for cropping images
        Available options:
            batch = # of images for each batch
            size = target (square) size for cropped window
        """
        print("Initializing Batch Dataset Reader...")
        self.coco_ids = np.array(coco_ids)
        self.size = self.coco_ids.shape[0]
        self.coco_dir = coco_dir
        self.ct = ct
        if image_options is None:
            self.image_options = NO_BATCH_OPT
            self._read_images = self._simple_read
        else:
            self.image_options = image_options
            self._read_images = self._batch_read
        print(self.image_options)

        self.pre_saved = pre_saved
        if pre_saved:
            self._get_images = self._load_images
        else:
            self._get_images = self._gen_images

        self.images = None
        self.annotations = None
        self.weights = None
        self.batch_offset = 0
        self.epoch = 1

    def next_batch(self):
        batch_size = self.image_options['batch']

        if (self.batch_offset + batch_size) > self.size:
            # Epoch finished, shuffle filenames
            self.epoch += 1
            np.random.shuffle(self.coco_ids)
            # Start next epoch
            self.batch_offset = 0

        batch = slice(self.batch_offset, self.batch_offset+batch_size)
        self.batch_offset += batch_size

        return self._read_images(batch)


    def _batch_read(self, pos):
        """
        Read multiple images/annotations/weights and crop to same size.
        This should be used when random crops are to be used.
        :param pos: slice object
        """
        n = pos.stop - pos.start
        size = self.image_options['size']

        images = np.zeros((n, size, size, 3), dtype=np.uint8)
        annotations = np.zeros((n, size, size, 1), dtype=np.uint8)
        weights = np.zeros((n, size, size, 1), dtype=np.float32)
        coco_ids = np.zeros((n), dtype=object)

        for i, coco_id in enumerate(self.coco_ids[pos]):
            res = self._get_images(coco_id)
            if self.pre_saved:
                image, annotation, weight = res
            else:
                valid_anns = [
                    ann for ann in self.ct.imgToAnns[coco_id]
                    if self.ct.anns[ann]['legibility'] == 'legible'
                ]
                ann = np.random.choice(valid_anns)
                window = coco_utils.get_window(res[1].shape, self.ct.anns[ann])
                image, annotation, weight = coco_utils.crop_resize(res, window, size)
            images[i] = image
            annotations[i] = annotation[:,:,None]
            weights[i] = weight[:,:,None]
            coco_ids[i] = coco_id

        return images, annotations, weights, coco_ids

    def _simple_read(self, pos):
        """
        Read only 1 image/annotation/weight of any size.
        This should be used only when batch size is equal to 1 and no randomness
        on the dataset is allowed.
        :param pos: slice object of length 1
        """
        assert((pos.stop - pos.start) == 1)
        coco_id = self.coco_ids[pos][0]
        image, annotation, weight = self._get_images(coco_id)

        # Add batch_dim + convert to floating point in [0,1]
        image = np.expand_dims(image, axis=0)
        # [None,:,:,None] ==> expand_dims(_, axis=[0,3]) // syntax not supported
        annotation = annotation[None,:,:,None]
        weight = weight[None,:,:,None]
        coco_id = np.array([coco_id])

        return image, annotation, weight, coco_id

    def _gen_images(self, coco_id):
        """
        Generate images using self.ct datas
        :param coco_id: image's coco id
        :return: image, its groundtruth w/o illegibles and its weights
        """
        fname = self.ct.imgs[coco_id]['file_name']
        image = cv2.imread(
            os.path.join(self.coco_dir, 'images/', fname),
        )
        annotation = np.zeros(image.shape[:-1], dtype=np.uint8)
        weight = np.ones(image.shape[:-1], np.float32)

        for ann in self.ct.imgToAnns[coco_id]:
            poly = np.array(self.ct.anns[ann]['polygon'], np.int32).reshape((4,2))

            if self.ct.anns[ann]['legibility'] == 'legible':
                # draw only legible bbox/polygon
                cv2.fillConvexPoly(annotation, poly, 255)
            else:
                # 0 weight if it is illegible
                cv2.fillConvexPoly(weight, poly, 0.0)

        return [image, annotation, weight]

    def _load_images(self, coco_id):
        """
        Load images already saved on the disk
        """
        fname = 'COCO_train2014_%012d.png' % coco_id
        image = cv2.imread(
            os.path.join(self.coco_dir, 'subset_validation/images/', fname))
        ann = cv2.imread(
            os.path.join(self.coco_dir, 'subset_validation/anns/', fname))
        ann = cv2.cvtColor(ann, cv2.COLOR_BGR2GRAY)
        weight = cv2.imread(
            os.path.join(self.coco_dir, 'subset_validation/weights', fname))
        weight = cv2.cvtColor(weight, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.

        return [image, ann, weight]