from __future__ import print_function
from six.moves import xrange

import argparse
import tensorflow as tf

import BatchDatasetReader as dataset
import coco_utils
from FCN_net import FCN

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default='1e-04', help='learning rate for Adam Optimizer')
parser.add_argument('--image_size', type=int, default=256, help='image size for training')
parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
parser.add_argument('--keep_prob', type=float, default=0.85, help='keep probability with dropout')
parser.add_argument('--logs_dir', default='logs/', help='path to logs directory')
parser.add_argument('--coco_dir', default='COCO_Text/', help='path to dataset')
parser.add_argument('--mode', required=True, choices=['train', 'val', 'test'])
args = parser.parse_args()


if __name__ == '__main__':
    if args.logs_dir is not None:
        ckpt = tf.train.get_checkpoint_state(args.logs_dir)

    fcn = FCN(2, args.logs_dir, lr=args.learning_rate, checkpoint=ckpt)

    train, val, test = coco_utils.read_dataset(args.coco_dir)

    if args.mode == 'train':
        opt = {
            'batch': args.batch_size,
            'smin': 192,
            'smax': 512,
            'size': args.image_size
        }
        
        train_set = dataset.BatchDataset(train, args.coco_dir, opt)
        # We want to keep track of validation loss on a constant dataset
        # => no crops
        opt = {
            'batch': 1,
            'smin': 192,
            'smax': 512,
            'size': 0
        }
        val_set = dataset.BatchDataset(val, args.coco_dir, opt)
        # We pass val set to keep track of its loss
        fcn.train(train_set, val_set=val_set, keep_prob=args.keep_prob)
        
    elif args.mode == 'val':
        opt = {
            'batch': args.batch_size,
            'smin': 192,
            'smax': 512,
            'size': args.image_size
        }
        val_set = dataset.BatchDataset(val, args.coco_dir, opt)
        FCN.validate(val_set)

    elif args.mode == 'test':
        # We just pass the filenames
        FCN.test(test)



	
