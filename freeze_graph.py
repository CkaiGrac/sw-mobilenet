#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
from mobilenet import Mobilenet

pb_file = "./mobilenet_normal.pb"
ckpt_file = "./ckpt/mobileNet_test_acc=0.7056.ckpt"
output_node_names = ["MobileNet/prediction"]


saver = tf.train.import_meta_graph(
    './ckpt/mobileNet_test_acc=0.7056.ckpt.meta', clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()

with tf.Session() as sess:
    saver.restore(sess, ckpt_file)
    output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
        sess=sess,
        input_graph_def=input_graph_def,  # 等于:sess.graph_def
        output_node_names=output_node_names)  # 如果有多个输出节点，以逗号隔开

    with tf.gfile.GFile(pb_file, "wb") as f:  # 保存模型
        f.write(output_graph_def.SerializeToString())  # 序列化输出
    print("%d ops in the final graph." %
          len(output_graph_def.node))  # 得到当前图有几个操作节点
