# Autograd tutorial from:
#   https://github.com/HIPS/autograd/blob/master/docs/tutorial.md
#  

import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad, elementwise_grad

def taylor_sine(x):  # Taylor approximation to sine function
  ans = currterm = x
  i = 0
  while np.abs(currterm) > 0.001:
    currterm = -currterm * x**2 / ((2 * i + 3) * (2 * i + 2))
    ans = ans + currterm
    i += 1
  return ans

def exp(x):
  """
  inputs:
    x     1d array of floats

  returns
    exp(x)  1d array of floats - exponentiation
  """
  return np.exp(x)

grad_sine = grad(taylor_sine)
print "Gradient of sin(pi) is", grad_sine(np.pi)


x = np.random.rand(10)
print x
print exp(x)

grad_exp = elementwise_grad(exp)
print 'Gradients of exp(x) is', grad_exp(x)
