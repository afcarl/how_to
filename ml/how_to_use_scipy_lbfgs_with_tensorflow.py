# How to use scipy's l-bfgs with tensorflow
# 
# The motivations are
# 1) tensorflow does not have native l-bfgs optimizer.
# 2) sometimes we want to include complex gradient/hessian operations in our computation, 
#       this can be done using tf.gradients() easily.
# 
# 
# some other tools may also be able to complete the task but I am more comfortable with tensorflow
# I found that autograd can also with work with scipy's l-bfgs (by computing the gradients automatically)


# some problems I encountered:
# 1) scipy's l-bfgs only works with float64 (incompatible with the default tf.float32!)
# 2) setting default session using tf.InteractiveSession()
# 3) the output of the grad function must be an np.array rather than a list of floats

import tensorflow as tf
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b


''' tensorflow functions '''
def func(xy):
  x = tf.placeholder(tf.float64)
  y = tf.placeholder(tf.float64)
  f = tf.pow(x, 2) + tf.pow(y,2)
  return f.eval(feed_dict={x:xy[0], y:xy[1]})

def func_grad(xy):
  x = tf.placeholder(tf.float64)
  y = tf.placeholder(tf.float64)
  f = tf.pow(x, 2) + tf.pow(y,2)
  grad = tf.pack([tf.gradients(f, x)[0], tf.gradients(f, y)[0]])
  return grad.eval(feed_dict={x:xy[0], y:xy[1]})

''' regular function counter-parts '''
def func1(xy):
  return xy[0]**2 + xy[1]**2

def func1_grad(xy):
  # note here the grad output must be an np.array rather than a list to work with the optimizer
  return np.array([2*xy[0], 2*xy[1]]) 


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
initial_values = np.array([100.0, 100.0])
  
print func1([1,1]), func1_grad([1,1])
print func([1,1]), func_grad([1,1])

print fmin_l_bfgs_b(func1, x0=initial_values, fprime=func1_grad)
print fmin_l_bfgs_b(func, x0=initial_values, approx_grad=True)
print fmin_l_bfgs_b(func, x0=initial_values, fprime=func_grad)
  

''' autograd example'''
# from autograd import grad, elementwise_grad
# grad = elementwise_grad(func1)
# print grad([10.0,10.0])
# print fmin_l_bfgs_b(func1, x0=initial_values, fprime=grad)

