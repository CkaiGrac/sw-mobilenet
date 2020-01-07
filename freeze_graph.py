#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
from mobilenet import Mobilenet

pb_file = "./mobilenet_normal.pb"
ckpt_file = "./ckpt/mobileNet_test_acc=0.7056.ckpt"
output_node_names = ["input/input_data", "transfer_learning/prediction", "transfer_learning/sqz"]

with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32, shape=(None,224,224,3), name='input_data')

sqz,predictions = Mobilenet(input_data,trainable=False)

sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                            input_graph_def  = sess.graph.as_graph_def(),
                            output_node_names = output_node_names)

with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())




