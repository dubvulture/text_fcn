# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import os

import cv2
import numpy as np

from text_fcn import coco_utils
from text_fcn.dataset_reader.dataset_reader import BatchDataset


class CocoDataset(BatchDataset):

    def __init__(self,
                 coco_ids,
                 ct,
                 coco_dir,
                 batch_size,
                 crop_size=0,
                 pre_saved=False):
        """
        :param coco_ids:
        :param batch_size:
        :param ct: COCO_Text object instance
        :param coco_dir: directory to coco dataset
        :param crop_size: window cropping size on each image (0 if none)
        :param pre_saved: whether to read images from storage or generate them on the go
        """
        # crop only when crop_size if given AND images are not loaded from disk
        crop_fun = self._crop_resize if crop_size > 0 and not pre_saved else None
        BatchDataset.__init__(self, coco_ids, batch_size, crop_size, image_op=crop_fun)

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
        fname = self.ct.imgs[coco_id]['file_name']
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
        intersected = []
        from shapely.geometry import box as Box
        from shapely.geometry import MultiPolygon, Polygon
        boxes = np.array([np.array(self.ct.anns[ann]['bbox']) for ann in valid_anns])
        boxes[:,2:] += boxes[:,:2]
        polygons = [Box(*poly) for poly in boxes]

        def draw_separation():
            for i, poly1 in enumerate(polygons):
                for j, poly2 in enumerate(polygons):
                    if poly1.contains(poly2) or poly2.contains(poly1):
                        intersected.append((i, j))
                        intersected.append((j, i))
                        continue
                    if (poly1 is not poly2
                        and (i, j) not in intersected
                        and poly1.intersects(poly2)):
                        # Intersect and draw line
                        intersected.append((i, j))
                        intersected.append((j, i))
                        union = poly1.union(poly2)
                        inter = poly1.intersection(poly2)
                        if isinstance(union, MultiPolygon):
                            continue
                        verteces = np.transpose(union.exterior.coords.xy)[:-1]
                        points = []
                        for i in range(verteces.shape[0]):
                            [x1, y1], [x2, y2], [x3, y3] = verteces[i-1],\
                                                           verteces[i],\
                                                           verteces[(i+1) % (verteces.shape[0])]
                            if (x2 - x1)*(y3 - y2) - (x3 - x2)*(y2 - y1) > 0:
                                # concave
                                points.append(verteces[i])

                        if len(points) == 4:
                            continue
                        elif len(points) == 1:
                            inter = inter.exterior if isinstance(inter, Polygon) else inter
                            coords = inter.coords.xy
                            coords = np.transpose(coords)
                            pos = np.argwhere(np.all(coords == points[0], axis=1))[0,0]
                            opposite = coords[(pos + 1) % coords.shape[0]]
                            points.append(opposite)
                        elif len(points) == 0:
                             #compute points
                            continue

                        assert len(points) == 2
                        width = 5
                        cv2.line(annotation, tuple(points[0].astype(np.int32)),
                                 tuple(points[1].astype(np.int32)), 127, width)

        max_expansion = 10
        for _ in range(max_expansion):
            draw_separation()
            for i, box in enumerate(boxes):
                boxes[i] += [-1, -1, 1, 1]
                polygons[i] = Box(*box)


        return [image, annotation, weight]

    def _load_image(self, coco_id):
        """
        Load image already saved on the disk
        """
        fname = 'COCO_train2014_%012d.png' % coco_id
        image = cv2.imread(
            os.path.join(self.coco_dir, 'subset_validation/images/', fname))
        annotation = cv2.imread(
            os.path.join(self.coco_dir, 'subset_validation/anns/', fname))
        annotation = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY)
        weight = cv2.imread(
            os.path.join(self.coco_dir, 'subset_validation/weights', fname))
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
        return coco_utils.crop_resize([image, annotation, weight], window, self.crop_size)
