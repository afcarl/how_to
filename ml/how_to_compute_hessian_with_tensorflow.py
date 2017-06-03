# How to compute gradients and hessian matrix using tensorflow

import tensorflow as tf
import numpy as np

x = tf.Variable(np.random.random_sample(), dtype=tf.float32)
y = tf.Variable(np.random.random_sample(), dtype=tf.float32)

f = tf.pow(x, 2) + 2 * x * y + 3 * tf.pow(y, 2) + \
    4 * x + 5 * y + 6 + x * x * y

# credit:
#   https://stackoverflow.com/questions/35266370/tensorflow-compute-hessian-matrix-and-higher-order-derivatives
def compute_hessian(fn, vars):
  # arg1: our defined function, arg2: list of tf variables associated with the function
  H = []
  for v1 in vars:
    temp = []
    for v2 in vars:
      # computing derivative twice, first w.r.t v2 and then w.r.t v1
      temp.append(tf.gradients(tf.gradients(fn, v2)[0], v1)[0])
    # tensorflow returns None when there is no gradient, so we replace None with 0
    temp = [0 if t == None else t for t in temp] 
    temp = tf.pack(temp)
    H.append(temp)
  H = tf.pack(H)
  return H.eval()

def compute_grad(fn, vars):
  grad = []
  for v in vars:
    grad.append(tf.gradients(fn, v)[0])
  grad = [0 if g == None else g for g in grad]
  grad = tf.pack(grad)
  return grad.eval()


sess = tf.Session()
sess.run(tf.global_variables_initializer())
with sess.as_default():
  print compute_hessian(f, [x, y])
  print compute_grad(f, [x, y])

