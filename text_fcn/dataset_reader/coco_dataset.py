# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import os

import cv2
import numpy as np
from shapely.geometry import box as Box
from shapely.geometry import LineString
from shapely.geometry import MultiPolygon

from text_fcn import coco_utils
from text_fcn.dataset_reader.dataset_reader import BatchDataset


class CocoDataset(BatchDataset):

    def __init__(self,
                 coco_ids,
                 ct,
                 coco_dir,
                 batch_size,
                 image_size,
                 crop=False,
                 pre_saved=False,
                 augment_data=True):
        """
        :param coco_ids:
        :param ct: COCO_Text object instance
        :param coco_dir: directory to coco dataset
        :param batch_size:
        :param image_size: crop window size if pre_saved=False
                           image size if pre_saved=True (0 if variable)
        :param crop: whether to crop images to image_size
        :param pre_saved: whether to read images from storage or generate them on the go
        :param augment_data: wheter to rotate&shuffle_bgr images

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
        BatchDataset.__init__(self,
                              coco_ids,
                              batch_size,
                              image_size,
                              image_op=crop_fun,
                              augment_data=augment_data)

        self.ct = ct
        self.coco_dir = coco_dir
        self.pre_saved = pre_saved

        if self.pre_saved:
            self._get_image = self._load_image
        else:
            self._get_image = self._gen_image

    def _gen_image(self, coco_id):
        """
        Generate images using self.ct data
        :param coco_id: image's coco id
        :return: image, its groundtruth w/o illegibles and its weights
        """
        fname = self.ct.imgs[coco_id]['file_name'][:-3] + 'png'
        image = cv2.imread(
            os.path.join(self.coco_dir, 'images/', fname)
        )
        annotation = np.zeros(image.shape[:-1], dtype=np.uint8)
        weight = np.ones(annotation.shape, np.float32)

        for ann in self.ct.imgToAnns[coco_id]:
            poly = np.array(self.ct.anns[ann]['polygon'], np.int32).reshape((4,2))
            if self.ct.anns[ann]['legibility'] == 'legible':
                # draw only legible bbox/polygon
                cv2.fillConvexPoly(annotation, poly, 255)
            else:
                # 0 weight if it is illegible
                cv2.fillConvexPoly(weight, poly, 0.0)

        valid_anns = [
            ann
            for ann in self.ct.imgToAnns[coco_id]
            if self.ct.anns[ann]['legibility'] == 'legible'
        ]
        if len(valid_anns) > 0:
            boxes = np.array([np.float32(self.ct.anns[ann]['bbox']) for ann in valid_anns])
            boxes[:,2:] += boxes[:,:2]
            self._draw_separation(annotation, boxes, max_expansion=5)

        return [image, annotation, weight]

    def _load_image(self, coco_id):
        """
        Load image already saved on the disk
        """
        fname = 'COCO_train2014_%012d.png' % coco_id
        image = cv2.imread(
            os.path.join(self.coco_dir, 'images/', fname))
        annotation = cv2.imread(
            os.path.join(self.coco_dir, 'anns/', fname))
        annotation = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY)
        weight = cv2.imread(
            os.path.join(self.coco_dir, 'weights/', fname))
        weight = cv2.cvtColor(weight, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.

        return [image, annotation, weight]

    def _crop_resize(self, image, annotation, weight, name=None):
        # next level hacks
        assert name is not None
        valid_anns = [
            ann for ann in self.ct.imgToAnns[name]
            if self.ct.anns[ann]['legibility'] == 'legible'
        ]
        ann = np.random.choice(valid_anns)
        # extract bbox => x, y, w, h
        bbox_rect = np.int32(self.ct.anns[ann]['bbox'])
        window = coco_utils.get_window(annotation.shape[:2], bbox_rect)
        return coco_utils.crop_resize([image, annotation, weight], window, self.image_size)

    def _draw_separation(self, image, boxes, max_expansion):
        """
        Draw separation lines dividing each box from another (if certain conditions are met)
        :param image: where to draw
        :param boxes: numpy array of bounding boxes
        :param max_expansion: maximum number of iterations with expansion
        """
        intersected = []
        polygons = [Box(*poly) for poly in boxes]

        def check():
            for i, poly1 in enumerate(polygons):
                for j, poly2 in enumerate(polygons):
                    if any(tpl in intersected for tpl in [(i, j), (j, i)]):
                        continue
                    line = self._get_separation(poly1, poly2)
                    if line is None:
                        continue
                    assert len(line) == 2
                    intersected.append((i,j))
                    intersected.append((j,i))
                    cv2.line(image,
                             tuple(line[0].astype(np.int32)),
                             tuple(line[1].astype(np.int32)),
                             color=127,
                             thickness=5)
        check()
        for _ in range(max_expansion):
            check()
            for i, box in enumerate(boxes):
                # expand bounding boxes w/o breaking their ratio
                side = boxes[i,:2] - boxes[i,2:]
                if side[0] > side[1]:
                    ratio = side[1] / side[0]
                    boxes[i,:] += [-1, -ratio, 1, ratio]
                else:
                    ratio = side[0] / side[1]
                    boxes[i,:] += [-ratio, -1, ratio, 1]
                polygons[i] = Box(*box)

    def _get_separation(self, poly1, poly2):
        """
        Given two polygon return their separation segment's verteces
        :type poly1: shapely.geometry.Polygon
        :type poly2: shapely.geometry.Polygon
        :return: numpy array with shape (2,2) representing the segment
        """
        if (poly1 is poly2
                or not poly1.intersects(poly2)
                or poly1.contains(poly2) or poly2.contains(poly1)):
            # skip cases
            return None

        union = poly1.union(poly2)
        if isinstance(union, MultiPolygon):
            # two verteces of the polygons touches
            return None
        inter = poly1.intersection(poly2)
        if inter.area >= 0.4 * min(poly1.area, poly2.area):
            # skip drawing if intersection area is too big
            return None

        verteces = np.transpose(union.exterior.coords.xy)[:-1]
        points = []
        if isinstance(inter, LineString):
            points = np.transpose(inter.coords.xy)
        else:
            for i in range(verteces.shape[0]):
                # get concave angles' vertices that will define the separation line
                [x1, y1], [x2, y2], [x3, y3] = verteces[i-1],\
                                               verteces[i],\
                                               verteces[(i+1) % (verteces.shape[0])]
                if (x2 - x1)*(y3 - y2) - (x3 - x2)*(y2 - y1) > 0:
                    # concave
                    points.append(verteces[i])

            if len(points) == 4:
                # Cross-shaped union (not clear how the division should work)
                points = None
            elif len(points) == 1:
                # L-shaped union (get the opposite vertex of the intersection)
                coords = np.transpose(inter.exterior.coords.xy)[:-1]
                pos = np.argwhere(np.all(coords == points[0], axis=1))[0,0]
                opposite = coords[(pos + 1) % coords.shape[0]]
                points.append(opposite)
            elif len(points) == 0:
                # Words on the same level (get fitting line segment of intersection)
                coords = np.transpose(inter.exterior.coords.xy)[:-1]
                points = [s1, s2, e1, e2] = np.array([
                    coords[0] - coords[1],
                    coords[1] - coords[2],
                    coords[2] - coords[3],
                    coords[3] - coords[0]
                ])
                arg = np.argmax([e1 - s1, e2 - s2])
                points = points[[arg, arg+2]]

        return points
