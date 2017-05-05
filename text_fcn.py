from __future__ import print_function
from six.moves import xrange

import os

import cv2
import tensorflow as tf
import numpy as np

import coco_utils
from networks.fcn import create_fcn
import tf_utils



class text_fcn(object):

    def __init__(self,
                 logs_dir,
                 lr=1e-04,
                 checkpoint=None,
                 train_freq=10,
                 val_freq=0,
                 save_freq=500):
        """
        :param logs_dir: directory for logs
        :param lr: initial learning rate
        :param checkpoint: a CheckpointState from get_checkpoint_state
        :param train_freq: trace train_loss every train_freq
        :param val_freq: trace val_loss every val_freq
        :param save_freq: save model every save_freq
        """
        self.logs_dir = logs_dir
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.image = tf.placeholder(
            tf.float32, shape=[None, None, None, 3], name='image')
        self.annotation = tf.placeholder(
            tf.int32, shape=[None, None, None, 1], name='annotation')
        self.weight = tf.placeholder(
            tf.float32, shape=[None, None, None, 1], name='weight')

        self.prediction, self.logits = create_fcn(self.image, self.keep_prob, 2)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.loss_op = self._loss()
        self.train_op = self._training(lr, global_step)

        tf.summary.scalar('train_loss', self.loss_op, collections=['train'])
        tf.summary.scalar('val_loss', self.loss_op, collections=['val'])

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var, collections=['train'])

        self.summ_train = tf.summary.merge_all(key='train') 
        self.summ_val = tf.summary.merge_all(key='val') 

        self.sv = self._setup_supervisor(checkpoint)

        self.train_freq = train_freq
        self.save_freq = save_freq
        self.val_freq = val_freq

    def train(self, train_set, val_set=None, keep_prob=0.85, max_steps=0):
        """
        :param train_set: training set (cropped images of same size)
        :param val_set: validation set (to keep track of its loss)
        :param keep_prob: 1-dropout
        :param max_steps: max steps to perform
        """
        with self.sv.managed_session() as sess:
            while not self.sv.should_stop():
                images, anns, weights, _ = train_set.next_batch()
                feed = {
                    self.image: images,
                    self.annotation: anns,
                    self.weight: weights,
                    self.keep_prob: keep_prob
                }
                sess.run(self.train_op, feed_dict=feed)

                step = sess.run(self.sv.global_step)

                if (step == max_steps) or ((step % self.train_freq) == 0):
                    loss, summary = sess.run(
                        [self.loss_op, self.summ_train],
                        feed_dict=feed)   
                    self.sv.summary_computed(sess, summary, step)
                    print('Step %d\tTrain_loss: %g' % (step, loss))

                if ((step == max_steps) or ((val_set is not None) and
                                            (step % self.val_freq == 0)):
                    images, anns, weights, _ = val_set.next_batch()
                    feed = {
                        self.image: images,
                        self.annotation: anns,
                        self.weight: weights,
                        self.keep_prob: 1.0
                    }
                    # no backpropagation
                    loss, summary = sess.run(
                        [self.loss_op, self.summ_val],
                        feed_dict=feed)
                    self.sv.summary_computed(sess, summary, step)
                    print('Step %d\tValidation loss: %g' % (step, loss))

                if (step == max_steps) or ((step % self.save_freq) == 0):
                    self.sv.saver.save(sess, self.logs_dir + 'model.ckpt', step)
                    print('Step %d\tModel saved.' % step)

                if step == max_steps:
                    break

    def test(self, filenames, directory):
        """
        Run on images in directory without their groundtruth
        (hence this should be used only during validation/testing phase)
        :param filenames: 
        """
        with self.sv.managed_session() as sess:
            for i, fname in enumerate(filenames):
                in_path = os.path.join(directory, 'images/', fname)
                in_image = cv2.imread(in_path + '.jpg')
                in_image = np.expand_dims(in_image, axis=0)

                feed = {self.image: in_image, self.keep_prob: 1.0}
                pred = sess.run(self.prediction, feed_dict=feed)
                print('Evaluated image\t' + fname)

                output = np.squeeze(pred, axis=3)[0]
                tf_utils.save_image(
                    coco_utils.to_ann(output),
                    self.logs_dir,
                    name=fname + '_output')

    def visualize(self, vis_set):
        """
        Run on given images in order to save input & gt & prediction
        """
        with self.sv.managed_session() as sess:
            while True:
                images, anns, weights, coco_ids = vis_set.next_batch()

                if vis_set.epoch != 1:
                    # Just one iteration over all dataset's images
                    break

                feed = {
                    self.image: images,
                    self.annotation: anns,
                    self.keep_prob: 1.0
                }
                preds = sess.run(self.prediction, feed_dict=feed)
                preds = np.squeeze(preds, axis=3)
                anns = np.squeeze(anns, axis=3)

                for i in range(vis_set.image_options['batch']):
                    tf_utils.save_image(
                        (images[i] * 255).astype(np.uint8),
                        self.logs_dir,
                        name='input_%05d' % coco_ids[i])
                    tf_utils.save_image(
                        coco_utils.to_ann(anns[i]),
                        self.logs_dir,
                        name='gt_%05d' % coco_ids[i])
                    tf_utils.save_image(
                        coco_utils.to_ann(weights[i]),
                        self.logs_dir,
                        name='wt_%05d' % coco_ids[i])
                    tf_utils.save_image(
                        coco_utils.to_ann(preds[i]),
                        self.logs_dir,
                        name='pred_%05d' % coco_ids[i])

                    print('Saved image: %d' % coco_ids[i])


    def _training(self, lr, global_step):
        """
        Setup the training phase
        :param lr: initial learning rate
        :param global_step: global step of training
        """
        optimizer = tf.train.AdamOptimizer(lr)
        grads = optimizer.compute_gradients(self.loss_op)
        return optimizer.apply_gradients(grads, global_step=global_step)

    def _loss(self):
        """
        Setup the loss function
        """
        return tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(
                logits=self.logits,
                labels=self.annotation,
                weights=self.weight
            )
        )

    def _setup_supervisor(self, checkpoint):
        """
        Setup the summary writer and variables
        :param checkpoint: saved model if any
        """
        saver = tf.train.Saver(max_to_keep=10)
        sv = tf.train.Supervisor(
            logdir=self.logs_dir,
            save_summaries_secs=0,
            save_model_secs=0,
            saver=saver)

        # Restore checkpoint if given
        if checkpoint and checkpoint.model_checkpoint_path:
            print('Checkpoint found, loading model')
            with sv.managed_session() as sess:
                sv.saver.restore(sess, checkpoint.model_checkpoint_path)

        return sv
