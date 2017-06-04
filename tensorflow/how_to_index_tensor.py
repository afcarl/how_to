# How to do tensors indexing
import tensorflow as tf
sess = tf.Session()
foo = tf.constant([[1,2,3], [4,5,6]])
print sess.run(foo[:, 0])