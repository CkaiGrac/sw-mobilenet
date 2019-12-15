#!/usr/bin/env python
# -*- coding_utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


class Mobilenet:

    def __init__(self, input, trainable):
        self.input = input
        self.trainable = trainable
        self.outputs = self.__build_network()

    def hard_swish(self, x, name='hard_swish'):
        with tf.name_scope(name):
            h_swish = x*tf.nn.relu6(x+3)/6
            return h_swish

    def separable_conv_block(self, input, dw_filter, output_channel, strides, name):

        with tf.variable_scope(name):

            dw_weight = tf.get_variable(name='dw_filter', dtype=tf.float32, trainable=True,
                                        shape=dw_filter, initializer=tf.random_normal_initializer(stddev=0.01))

            dw = tf.nn.depthwise_conv2d(
                input=input, filter=dw_weight, strides=strides, padding="SAME", name='Conv/dw')

            bn_dw = tf.layers.batch_normalization(dw, beta_initializer=tf.zeros_initializer(),
                                                  gamma_initializer=tf.ones_initializer(),
                                                  moving_mean_initializer=tf.zeros_initializer(),
                                                  moving_variance_initializer=tf.ones_initializer(), training=self.trainable,
                                                  name='dw/bn')
            # relu = tf.nn.leaky_relu(bn_dw, 0.1)
            # h_swish = self.hard_swish(bn_dw)
            weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                     shape=(1, 1, dw_filter[2]*dw_filter[3], output_channel), initializer=tf.random_normal_initializer(stddev=0.01))

            conv = tf.nn.conv2d(input=bn_dw, filter=weight, strides=[
                                1, 1, 1, 1], padding="SAME", name="conv/s1")
            bn_pt = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                  gamma_initializer=tf.ones_initializer(),
                                                  moving_mean_initializer=tf.zeros_initializer(),
                                                  moving_variance_initializer=tf.ones_initializer(),
                                                  training=self.trainable,
                                                  name='pt/bn')
            # return tf.nn.leaky_relu(bn_pt, 0.1)
            if output_channel >= 512:
                return self.hard_swish(bn_pt)
            else:
                return tf.nn.leaky_relu(bn_pt, 0.1)

    def __build_network(self):

        with tf.variable_scope('MobileNet'):
            conv1 = tf.layers.conv2d(self.input,
                                     filters=32,
                                     kernel_size=(3, 3),
                                     strides=(2, 2),
                                     padding='same',
                                     activation=tf.nn.relu6,   # tf.nn.relu,
                                     name='conv1'
                                     )
            bn1 = tf.layers.batch_normalization(conv1, beta_initializer=tf.zeros_initializer(),
                                                gamma_initializer=tf.ones_initializer(),
                                                moving_mean_initializer=tf.zeros_initializer(),
                                                moving_variance_initializer=tf.ones_initializer(), training=self.trainable,
                                                name='bn')
            x = self.separable_conv_block(input=bn1, dw_filter=(3, 3, 32, 1), output_channel=64,
                                          strides=(1, 1, 1, 1), name="spearable_1")

            x = self.separable_conv_block(input=x, dw_filter=(3, 3, 64, 1), output_channel=128,
                                          strides=(1, 2, 2, 1), name="spearable_2")

            x = self.separable_conv_block(input=x, dw_filter=(3, 3, 128, 1), output_channel=128,
                                          strides=(1, 1, 1, 1), name="spearable_3")

            x = self.separable_conv_block(input=x, dw_filter=(3, 3, 128, 1), output_channel=256,
                                          strides=(1, 2, 2, 1), name="spearable_4")

            x = self.separable_conv_block(input=x, dw_filter=(3, 3, 256, 1), output_channel=256,
                                          strides=(1, 1, 1, 1), name="spearable_5")
            route1 = x
            # opti
            x = self.separable_conv_block(input=x, dw_filter=(3, 3, 256, 1), output_channel=512,
                                          strides=(1, 2, 2, 1), name="spearable_6")

            x = self.separable_conv_block(input=x, dw_filter=(3, 3, 512, 1), output_channel=512,
                                          strides=(1, 1, 1, 1), name="spearable_7")

            x = self.separable_conv_block(input=x, dw_filter=(3, 3, 512, 1), output_channel=512,
                                          strides=(1, 1, 1, 1), name="spearable_8")

            x = self.separable_conv_block(input=x, dw_filter=(3, 3, 512, 1), output_channel=512,
                                          strides=(1, 1, 1, 1), name="spearable_9")

            x = self.separable_conv_block(input=x, dw_filter=(3, 3, 512, 1), output_channel=512,
                                          strides=(1, 1, 1, 1), name="spearable_10")

            x = self.separable_conv_block(input=x, dw_filter=(3, 3, 512, 1), output_channel=512,
                                          strides=(1, 1, 1, 1), name="spearable_11")
            route2 = x
            x = self.separable_conv_block(input=x, dw_filter=(3, 3, 512, 1), output_channel=1024,
                                          strides=(1, 1, 1, 1), name="spearable_12")

            x = self.separable_conv_block(input=x, dw_filter=(3, 3, 1024, 1), output_channel=1024,
                                          strides=(1, 1, 1, 1), name="spearable_13")
        return route1, route2, x


if __name__ == '__main__':
    input = tf.placeholder(dtype=tf.float32, shape=(
        None, 416, 416, 3), name='input')
    model = Mobilenet(input, True)
