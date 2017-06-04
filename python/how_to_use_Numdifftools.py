import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
x = np.linspace(-2, 2, 100)
for i in range(10):
  df = nd.Derivative(np.tanh, n=i)
  y = df(x)
  h = plt.plot(x, y/np.abs(y).max())
plt.show()

def rosen(x): 
  return (1-x[0])**2 + 105.*(x[1]-x[0]**2)**2

hessian = nd.Hessian(rosen)
print hessian([1,1])


