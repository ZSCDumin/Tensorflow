import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
# #
# result = v1 + v2
# init_op = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     graph_def = tf.get_default_graph().as_graph_def()
#
#     output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
#
#     with tf.gfile.GFile("model/combined_model.pb", "wb") as f:
#         f.write(output_graph_def.SerializeToString())


# with tf.Session() as sess:
#     with gfile.FastGFile("model/combined_model.pb", 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         result = tf.import_graph_def(graph_def, return_elements=["add:0"])
#         print(sess.run(result))

# saver = tf.train.Saver()
# saver.export_meta_graph("model/ckpt.meda.json", as_text=True)

reader = tf.train.NewCheckpointReader('model/model.ckpt')

all_variables = reader.get_variable_to_shape_map()

for variable_name in all_variables:
    print(variable_name, all_variables[variable_name])

print("Values for variable v1 is ",reader.get_tensor("v1"))
