# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import os

import cv2
import numpy as np

from text_fcn import coco_utils
from text_fcn.dataset_reader.dataset_reader import BatchDataset


class SynthDataset(BatchDataset):

        def __init__(self,
                     fnames,
                     st,
                     synth_dir,
                     batch_size,
                     image_size,
                     crop=False,
                     pre_saved=False):
            """
            :param fnames:
            :param st:
            :param synth_dir:
            :param batch_size:
            :param image_size: crop window size if pre_saved=False
                               image size if pre_saved=True (0 if variable)
            :param crop: whether to crop images to image_size
            :param pre_saved: whether to read images from storage or generate them on the go

            Here's some examples
                - pre_saved = True
                    - batch_size = 1, image_size = 0, crop = False
                        Load images from storage and do not crop them.
                    - batch_size = X, image_size = Y, crop = False
                        Load images from storage which are asserted to have the same size = image_size.
                    - batch_size = X, image_size = Y, crop = True
                        Load images from storage and crop them to image_size.
                - pre_saved = False
                    - batch_size = 1, image_size = 0, crop = False
                        Generate images and do not crop them.
                    - batch_size = X, image_size = Y, crop = False
                        Generate images which are asserted to have the same size = image_size.
                    - batch_size = X, image_size = Y, crop = True
                        Generate images and crop them to image_size.
            """
            # crop only when crop_size if given AND images are not loaded from disk
            crop_fun = self._crop_resize if crop else None
            BatchDataset.__init__(self, fnames, batch_size, image_size, image_op=crop_fun)

            self.st = st
            self.synth_dir = synth_dir
            self.pre_saved = pre_saved

            if self.pre_saved:
                self._get_image = self._load_image
            else:
                self._get_image = self._gen_image

        def _gen_image(self, fname):
            image = cv2.imread(os.path.join(self.synth_dir, 'images/', fname+'.jpg'))
            annotation = np.zeros(image.shape[:-1], dtype=np.uint8)
            weight = np.ones(image.shape[:-1], dtype=np.float32)

            for charBB in self.st[fname]['chars']:
                cv2.fillConvexPoly(annotation, charBB.astype(np.int32), 255)

            for wordBB in self.st[fname]['words']:
                thick = int(max(2, 2))
                cv2.drawContours(annotation,
                                 wordBB.reshape((1,4,1,2)).astype(np.int32),
                                 -1,
                                 127,
                                 thickness=thick)

            return [image, annotation, weight]

        def _load_image(self, fname):
            fname += '.png'
            images = cv2.imread(
                os.path.join(self.synth_dir, 'subset_validation/images/', fname))
            annotation = cv2.imread(
                os.path.join(self.synth_dir, 'subset_validation/anns/', fname))
            annotation = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY)
            weight = np.ones(annotation.shape, np.float32)

            return [images, annotation, weight]

        def _crop_resize(self, image, annotation, weight, name=None):
            assert name is not None
            choice = np.random.randint(0, self.st[name]['anns'].shape[0])
            bbox = np.floor(self.st[name]['anns'][choice])
            # [xi, yi : i=0..4] => x, y, w, h
            bbox = cv2.boundingRect(np.expand_dims(bbox, axis=1))
            bbox = np.int32(bbox)
            window = coco_utils.get_window(annotation.shape, bbox)
            return coco_utils.crop_resize([image, annotation, weight], window, self.image_size)