from __future__ import division, print_function
from six.moves import xrange

import argparse
from datetime import datetime
import json
import os
import subprocess

import cv2
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.measurements import label
from scipy.ndimage.measurements import labeled_comprehension as extract_feature
from scipy.ndimage.morphology import binary_closing as closing
import tensorflow as tf

from coco_text import COCO_Text, coco_evaluation
import coco_utils
from dataset_reader import BatchDataset
from text_fcn import text_fcn


parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default='1e-04', help='learning rate for Adam Optimizer')
parser.add_argument('--image_size', type=int, default=256, help='image size for training')
parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
parser.add_argument('--max_steps', type=int, default=0, help='max steps to perform, 0 for infinity')
parser.add_argument('--keep_prob', type=float, default=0.85, help='keep probability with dropout')
parser.add_argument('--logs_dir', default='logs/temp/', help='path to logs directory')
parser.add_argument('--coco_dir', default='COCO_Text/', help='path to dataset')
parser.add_argument('--mode', required=True, choices=['train', 'test', 'visualize'])
parser.add_argument('--save_freq', type=int, default=500, help='save model every save_freq')
parser.add_argument('--train_freq', type=int, default=20, help='trace train_loss every train_freq')
parser.add_argument('--val_freq', type=int, default=0, help='trace val_loss every val_freq')
parser.add_argument('--id_list', help='text file containing images\' coco ids to visualize')

args = parser.parse_args()

if args.id_list == None and args.mode == 'visualize':
    parser.error('--mode="visualize" requires --id_list')


def save_run():
    """
    Append current run git revision + python arguments to exp.lisp
    :return: git revision string
    """
    git_rev = subprocess.Popen(
        'git rev-parse --short HEAD',
        shell=True,
        stdout=subprocess.PIPE
    ).stdout.read()[:-1]

    print('Saving run to runs.sh - %s' % git_rev)
    with open('runs.sh', 'a') as f:
        f.write(datetime.now().strftime('# %H:%M - %d %b %y\n'))
        f.write('# git checkout %s\n' % git_rev)
        f.write('# python main.py ')
        for k, v in args._get_kwargs():
            f.write('--%s=\'%s\' ' % (k, v))
        f.write('\n# git checkout master\n\n')


if __name__ == '__main__':
    if not os.path.exists(args.coco_dir):
        raise Exception("coco_dir does not exist")

    args.coco_dir = os.path.abspath(args.coco_dir) + '/'
    args.logs_dir = os.path.abspath(args.logs_dir) + '/'
    
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
        ckpt = None
    else:
        # Get checkpoint from logs_dir if any
        ckpt = tf.train.get_checkpoint_state(args.logs_dir)

    print("Setting up FCN...")
    fcn = text_fcn(
        logs_dir=args.logs_dir,
        lr=args.learning_rate,
        checkpoint=ckpt,
        train_freq=args.train_freq,
        val_freq=args.val_freq,
        save_freq=args.save_freq)

    coco_utils.maybe_download_and_extract(args.coco_dir, coco_utils.URL, is_zipfile=True)
    coco_text = COCO_Text(os.path.join(args.coco_dir, 'COCO_Text.json'))
    train, val, test = coco_utils.read_dataset(coco_text)

    if args.mode == 'train':
        save_run()
        opt = {
            'batch': args.batch_size,
            'size': args.image_size
        }
        train_set = BatchDataset(train, args.coco_dir, coco_text, opt)
        # We want to keep track of validation loss on an almost constant dataset
        # => load previously saved images/gt/weights
        val_set = None
        if args.val_freq > 0:
            val_set = BatchDataset(val, args.coco_dir, coco_text, opt, pre_saved=True)

        # We pass val_set to keep track of its loss
        fcn.train(train_set, val_set=val_set, keep_prob=args.keep_prob, max_steps=args.max_steps)
        
    elif args.mode == 'visualize':
        with open(args.id_list, 'rb') as f:
            ids = [int(line) for line in f if line.strip() != '']
        ids_set = BatchDataset(ids, args.coco_dir, coco_text, None)
        fcn.visualize(ids_set)

    elif args.mode == 'test':
        # We just pass the ids
        fcn.test(test, args.coco_dir)

    elif args.mode == 'coco':
        # After NN extract bboxes and evaluate with coco_text
        perm = np.random.ranint(0, len(val), size=[42])
        fcn.test(val[perm], args.coco_dir)
        coco_pipe(coco_text, fnames)




def coco_pipe(coco_text):
    """
    :param coco_text: COCO_Text instance
    """
    fnames = os.listdir(os.path.join(args.logs_dir, 'test/'))
    results = os.path.join(args.logs_dir, 'results.json')
    for fname in filenames:
        image = cv2.imread(fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image[image > 0] = 255 (this should not be needed)
        bboxes, scores = get_bboxes(image)
        coco_id = int(fname[20:-11])

        jsonarr = []

        for i in xrange(np.size(bboxes)):
            jsonarr.append({
                'utf8_string': "null",
                'image_id': coco_id,
                'bbox': bboxex[i],
                "score": scores[i]
            })

    ct_res = ct.loadRes(jsonarr)
    imgIds = [pred['image_id'] for pred in jsonarr]
    detections = coco_evaluation.getDetections(
        ct, ct_res, imgIds=imgIds, detection_threshold = 0.5)
    coco_evaluation.printDetailedResults(ct, ct_res, None, 'FCN')


def get_bboxes(image):
    """
    Return bounding boxes found and their accuracy score (TBD)
    :param image: B/W image (values in {0, 255})
    """
    MIN_AREA = 32
    X = 3
    DIL = (3,3)
    ONES = np.ones(DIL, dtype=np.uint8)

    # pad image in order to prevent closing "constriction"
    output = np.pad(image, (DIL, DIL), 'constant')
    output = closing(output, structure=ONES, iterations=3).astype(np.uint8)
    # remove padding
    output = output[X:-X, X:-X]
    labels, num = label(output, structure=ONES)
    areas = extract_feature(output, labels, range(1, num+1), np.sum, np.int32, 0)
 
    for i in xrange(num):
        if areas[i] < MIN_AREA:
            labels[labels == i+1] = 0
 
    objs = find_objects(labels)
     
    bboxes = np.array([
        (obj[1].start, obj[1].stop, obj[0].start, obj[0].stop)
        for obj in objs if obj is not None
    ])
    # count white pixels inside the current bbox
    area = lambda i, b: np.count_nonzero(i[b[0]:b[1], b[2]:b[3]])
    # score as foreground / bbox_area
    scores = np.array([
        area(image, bbox) / np.multiply(bbox[[3,1]] - bbox[[2,0]])
        for bbox in bboxes
    ])
     
    return bboxes, scores