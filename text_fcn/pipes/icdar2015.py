from six.moves import range

import os

import cv2
import numpy as np
from scipy.ndimage.measurements import label
from scipy.ndimage.measurements import labeled_comprehension as extract_feature
from scipy.ndimage.morphology import binary_dilation as dilation


def get_bboxes(image):
    """
    Return bounding boxes found and their accuracy score (TBD)
    :param image: B/W image (values in {0, 255})
    """
    MIN_AREA = 64
    X = 3
    DIL = (X, X)
    ONES = np.ones(DIL, dtype=np.uint8)
 
    # pad image in order to prevent closing "constriction"
    output = np.pad(image, (DIL, DIL), 'constant')
    output = dilation(output, structure=ONES, iterations=1).astype(np.uint8)
    # remove padding
    output = output[X:-X, X:-X]
    labels, num = label(output, structure=ONES)
    if num > 0:
        areas = extract_feature(output, labels, range(1, num+1), np.sum, np.int32, 0)
 
        for i in range(num):
            if areas[i] < MIN_AREA:
                output[output == i+1] = 0
 
        cnts, _ = cv2.findContours(output.copy(),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)[-2:]

        boxPoints = cv2.boxPoints if cv2.__version__[0] == '3' else cv2.cv.BoxPoints
        bboxes = np.array(
            [
                np.int32(boxPoints(cv2.minAreaRect(cnt))).ravel()
                for cnt in cnts
            ],
            dtype=np.int32
        )
 
        # count white pixels inside the current bbox
        area = lambda b: np.count_nonzero(image[b[2]:b[3], b[0]:b[1]])
        # score as foreground / bbox_area
        scores = [
            #area(bbox) / np.prod(bbox[[3,1]] - bbox[[2,0]])
            1
            for bbox in bboxes
        ]
    else:
        bboxes, scores = None, None
 
    return bboxes, scores
 
 
def icdar_pipe(in_dir):
    directory = os.path.join(in_dir, 'output/')
    fnames = os.listdir(directory)
 
    for fname in fnames:
        image = cv2.imread(os.path.join(directory, fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image[image > 0] = 255 (this should not be needed)
        bboxes, scores = get_bboxes(image)
 
        if bboxes is not None:
            out = os.path.join(directory, 'res_%s.txt' % fname[:-11])
            with open(out, 'w') as f:
                for i in range(bboxes.shape[0]):
                    f.write(('{},'*7 + '{}\r\n').format(*bboxes[i]))