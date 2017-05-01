from __future__ import print_function
from six.moves import xrange

import os

import cv2
import tensorflow as tf
import numpy as np

import coco_utils
import tf_utils
from vgg_net import VGG_NET



class FCN(object):

    def __init__(self,
                 classes,
                 logs_dir,
                 lr=1e-04,
                 checkpoint=None,
                 train_freq=10,
                 val_freq=500,
                 save_freq=500):
        """
        :param classes: # of classes
        :param logs_dir: directory for logs
        :param lr: initial learning rate
        :param checkpoint: a CheckpointState from get_checkpoint_state
        :param train_freq: trace train_loss every train_freq
        :param val_freq: trace val_loss every val_freq
        :param save_freq: save model every save_freq
        """
        self.classes = classes
        self.logs_dir = logs_dir
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.image = tf.placeholder(
            tf.float32, shape=[None, None, None, 3], name='image')
        self.annotation = tf.placeholder(
            tf.int32, shape=[None, None, None, 1], name='annotation')
        self.weight = tf.placeholder(
            tf.float32, shape=[None, None, None, 1], name='weight')

        self.prediction, self.logits = self._setup_net()

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

    def train(self, train_set, val_set=None, keep_prob=0.85):
        """
        :param train_set: training set (cropped images of same size)
        :param val_set: validation set (to keep track of its loss)
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

                if step % self.train_freq == 0:
                    loss, summary = sess.run(
                        [self.loss_op, self.summ_train],
                        feed_dict=feed)   
                    self.sv.summary_computed(sess, summary, step)
                    print('Step %d\tTrain_loss: %g' % (step, loss))

                if (val_set is not None) and (step % self.val_freq == 0):
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

                if step % self.save_freq == 0:
                    self.sv.saver.save(sess, self.logs_dir + 'model.ckpt', step)
                    print('Step %d\tModel saved.' % step)

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

    def _setup_net(self):
        """
        Setup the main conv/deconv network
        """
        with tf.variable_scope('inference'):
            vgg_net = VGG_NET(self.image).net
            conv_final = vgg_net['conv5_4']
            output = tf_utils.max_pool_2x2(conv_final)

            conv_shapes = [
                [7, 7, 512, 4096],
                [1, 1, 4096, 4096],
                [1, 1, 4096, self.classes]
            ]

            for i, conv_shape in enumerate(conv_shapes):
                W = tf_utils.weight_variable(
                    conv_shape, name='conv%d_w' % (i+6))
                b = tf_utils.bias_variable(
                    conv_shape[-1:], name='conv%d_b' % (i+6))
                output = tf_utils.conv2d_basic(output, W, b)
                if i<2:
                    output = tf.nn.relu(output, name='relu%d' % (i+6))
                    tf_utils.add_activation_summary(output, collections=['train'])
                    output = tf.nn.dropout(output, keep_prob=self.keep_prob)

            pool4 = vgg_net['pool4']
            pool3 = vgg_net['pool3']

            deconv_shapes = [
                tf.shape(pool4),
                tf.shape(pool3),
                tf.stack([
                    tf.shape(self.image)[0], tf.shape(self.image)[1],
                    tf.shape(self.image)[2], self.classes
                ])
            ]

            W_shapes = [
                [4, 4, pool4.get_shape()[3].value, self.classes],
                [4, 4, pool3.get_shape()[3].value, pool4.get_shape()[3].value],
                [16, 16, self.classes, pool3.get_shape()[3].value]
            ]

            b_shapes = [[shape[2]] for shape in W_shapes]

            strides = [2, 2, 8]

            for i in xrange(3):
                W = tf_utils.weight_variable(
                    W_shapes[i], name='deconv_%d_w' % (i+1))
                b = tf_utils.bias_variable(
                    b_shapes[i], name='deconv_%d_b' % (i+1))
                output = tf_utils.conv2d_transpose_strided(
                    output, W, b,
                    output_shape=deconv_shapes[i], stride=strides[i])
                if i<2:
                    output = tf.add(output, vgg_net['pool%d' % (4-i)])

            prediction = tf.argmax(output, dimension=3, name='prediction')
        return tf.expand_dims(prediction, dim=3), output

    def _setup_supervisor(self, checkpoint):
        """
        Setup the summary writer and variables
        :param checkpoint: saved model if any
        """
        saver = tf.train.Saver(max_to_keep=1)
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
