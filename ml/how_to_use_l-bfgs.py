# How to use scipy's l-bfgs
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b

# example 1:
#     credits: https://stackoverflow.com/questions/28256737/why-isnt-arange-defined
x_true = np.arange(0,10,0.1)
m_true = 2.5
b_true = 1.0
y_true = m_true*x_true + b_true

def func(params, *args):
    x = args[0]
    y = args[1]
    m, b = params
    y_model = m*x+b
    error = y-y_model
    return sum(error**2)

initial_values = np.array([1.0, 0.0])
mybounds = [(None,2), (None,None)]

print fmin_l_bfgs_b(func, x0=initial_values, args=(x_true, y_true), approx_grad=True)
print fmin_l_bfgs_b(func, x0=initial_values, args=(x_true, y_true), 
  bounds=mybounds, approx_grad=True)


# example 2:
#   credits: https://gist.github.com/yuyay/3067185
def rosen(x):
   return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:1])**2.0)

def rosen_der(x):
   xm = x[1:-1]
   xm_m1 = x[:-2]
   xm_p1 = x[2:]
   der = np.zeros_like(x)
   der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
   der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
   der[-1] = 200*(x[-1]-x[-2]**2)
   return der

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
print fmin_bfgs(rosen, x0, fprime=rosen_der)
print fmin_l_bfgs_b(rosen, x0, fprime=rosen_der)