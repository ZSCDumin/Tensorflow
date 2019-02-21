import tensorflow as tf
import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# g1 = tf.Graph()
# with g1.as_default():
#     v = tf.get_variable("v", shape=[1], initializer=tf.zeros_initializer())
# g2 = tf.Graph()
# with g2.as_default():
#     v = tf.get_variable("v", shape=[1], initializer=tf.ones_initializer())
# with tf.Session(graph=g1) as sess:
#     tf.global_variables_initializer().run()
#     with tf.variable_scope("", reuse=True):
#         print(sess.run(tf.get_variable("v")))
# with tf.Session(graph=g2) as sess:
#     tf.global_variables_initializer().run()
#     with tf.variable_scope("", reuse=True):
#         print(sess.run(tf.get_variable("v")))

# a = tf.constant([1.0, 2.0], name="a")
# b = tf.constant([2.0, 3.0], name="b")
# result = tf.add(a, b, name="add")
# with tf.Session() as sess:
#     tf.global_variables_initializer()
#     print(sess.run(result))

with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))

with tf.variable_scope("foo", reuse=False):
    v1 = tf.get_variable("v", [1])
    print(v1 == v)

if __name__ == '__main__':
    A = [[1, 3, 4, 5, 6]]
    B = [[1, 3, 4], [2, 4, 1]]

    # tf.argmax(y_, 1) 取的是行的最大值对应的下标的索引
    # tf.argmax(y_, 0) 取的是列的最大值对应的下标的索引
    with tf.Session() as sess:
        print(sess.run(tf.argmax(A, 1)))
        print(sess.run(tf.argmax(B, 0)))
