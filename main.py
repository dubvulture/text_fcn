from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import os
import subprocess

import dill
import numpy as np
import tensorflow as tf

from text_fcn import coco_utils
from text_fcn import TextFCN
from text_fcn.coco_text import COCO_Text
from text_fcn.dataset_reader import CocoDataset
from text_fcn.dataset_reader import IcdarDataset
from text_fcn.pipes import coco_pipe, icdar_pipe


parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default='1e-05', help='learning rate for the optimizer')
parser.add_argument('--image_size', type=int, default=256, help='image size for training')
parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
parser.add_argument('--max_steps', type=int, default=0, help='max steps to perform, 0 for infinity')
parser.add_argument('--keep_prob', type=float, default=0.85, help='keep probability with dropout')
parser.add_argument('--logs_dir', default='logs/temp/', help='path to logs directory')
parser.add_argument('--mode', required=True, choices=['train', 'test', 'visualize', 'coco', 'icdar2015', 'icdar2013'])
parser.add_argument('--save_freq', type=int, default=500, help='save model every save_freq')
parser.add_argument('--train_freq', type=int, default=20, help='trace train_loss every train_freq')
parser.add_argument('--val_freq', type=int, default=0, help='trace val_loss every val_freq')
parser.add_argument('--id_list', help='text file containing images\' coco ids to visualize')
parser.add_argument('--dataset', default='cocotext', choices=['cocotext', 'icdartext'], help='which dataset')
parser.add_argument('--test_folder', help='')

args = parser.parse_args()

if args.id_list is None and args.mode == 'visualize':
    parser.error('--mode="visualize" requires --id_list')

if args.test_folder is None and args.mode == 'test':
    parser.error('--mode="test" requires --test_folder')

assert args.image_size % 32 == 0,\
       'image size must be a multiple of 32'

if args.mode == 'coco':
    assert args.dataset == 'cocotext'


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
    if args.dataset == 'cocotext':
        Dataset = CocoDataset
        dataset_dir = 'COCO_Text/'
    else: # args.dataset == 'icdartext'
        Dataset = IcdarDataset
        dataset_dir = 'ICDAR2015/'

    dataset_dir = os.path.abspath(dataset_dir) + '/'
    args.logs_dir = os.path.abspath(args.logs_dir) + '/'

    if not os.path.exists(dataset_dir):
        raise Exception("dataset does not exist")

    train_set = None
    val_set = None
    
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
        ckpt = None
    else:
        # Get checkpoint from logs_dir if any
        ckpt = tf.train.get_checkpoint_state(args.logs_dir)
        # And restore train/val sets if they have been serialized
        train_pickle = os.path.join(args.logs_dir, 'train_set.pkl')
        if os.path.exists(train_pickle):
            train_set = dill.load(open(train_pickle, 'rb'))
        val_pickle = os.path.join(args.logs_dir, 'val_set.pkl')
        if os.path.exists(val_pickle):
            val_set = dill.load(open(val_pickle, 'rb'))

    print("Setting up FCN...")
    fcn = TextFCN(
        logs_dir=args.logs_dir,
        lr=args.learning_rate,
        checkpoint=ckpt,
        train_freq=args.train_freq,
        val_freq=args.val_freq,
        save_freq=args.save_freq)

    # Train + Val both with COCO_Text and Icdar2015
    # Visualize & Test & Coco require COCO_Text
    # Icdar is unbound
    if ((args.mode in ['train', 'test'] and args.dataset == 'cocotext')
        or (args.mode in ['visualize', 'coco', 'icdar2015', 'icdar2013'])):
        coco_utils.maybe_download_and_extract(dataset_dir, coco_utils.URL, is_zipfile=True)
        chosen_text = COCO_Text(os.path.join(dataset_dir, 'COCO_Text.json'))
        read_dataset = coco_utils.coco_read_dataset
    elif args.mode == 'train' and args.dataset == 'icdartext':
        # args.dataset_dir == 'Synth_Text/'
        chosen_text = np.load(os.path.join(dataset_dir, 'icdar2015.npy'))[()]
        read_dataset = coco_utils.icdar_read_dataset
    else:
        print('???')

    train, val, test = read_dataset(chosen_text)

    if args.mode == 'train':
        save_run()
        # Load from storage and crop them to image_size
        train_set = train_set or Dataset(train,
                                         chosen_text,
                                         os.path.join(dataset_dir, 'word_division_train/'),
                                         args.batch_size,
                                         args.image_size,
                                         crop=True,
                                         pre_saved=True,
                                         augment_data=True)
        # We want to keep track of validation loss on an almost constant dataset
        # => load previously saved images/gt/weights
        if args.val_freq > 0:
            if args.dataset == 'cocotext':
                subset = os.listdir(os.path.join(
                    dataset_dir, 'word_division_val/images'))
                subset = [int(i[15:-4]) for i in subset]
            else: # args.dataset == 'icdar_text'
                raise NotImplementedError
            # Load from storage already cropped to image_size
            val_set = val_set or Dataset(subset,
                                         chosen_text,
                                         os.path.join(dataset_dir, 'word_division_val/'),
                                         args.batch_size,
                                         args.image_size,
                                         crop=False,
                                         pre_saved=True,
                                         augment_data=False)

        # We pass val_set (if given) to keep track of its loss
        fcn.train(train_set,
                  val_set=val_set,
                  keep_prob=args.keep_prob,
                  max_steps=args.max_steps)

    elif args.mode == 'visualize':
        # Save to args.logs_dir/visualize input images + groundtruth + predictions + text heatmap
        # of images written in args.id_list
        with open(args.id_list, 'rb') as f:
            ids = [int(line) for line in f if line.strip() != '']
        ids_set = CocoDataset(ids,
                              chosen_text,
                              dataset_dir,
                              batch_size=1,
                              image_size=0,
                              crop=False,
                              pre_saved=False,
                              augment_data=False)
        fcn.visualize(ids_set)

    elif args.mode == 'test':
        test = os.listdir(args.test_folder)
        fcn.test(test, args.test_folder, ext='')

    elif args.mode == 'coco':
        # After NN extract bboxes and evaluate with coco_text
        val = [chosen_text.imgs[coco_id]['file_name'][:-4] for coco_id in val]
        fcn.test(val, os.path.join(dataset_dir, 'images/'), ext='.jpg')
        coco_pipe(chosen_text, args.logs_dir, mode='validation')

    elif args.mode == 'icdar2015':
        # After NN extract bboxes (orientated) and save for online evaluation
        val_dir = '/home/manuel/Downloads/ICDAR2015/test/'
        val = [i[:-4] for i in os.listdir(val_dir)]
        fcn.test(val, val_dir, ext='.jpg')
        icdar_pipe(args.logs_dir)

    elif args.mode == 'icdar2013':
        val_dir = '/home/manuel/text_fcn/ICDAR2013/test/'
        val = [i[:-4] for i in os.listdir(val_dir)]
        fcn.test(val, val_dir, ext='.jpg')
        icdar_pipe(args.logs_dir, rotated=False)
