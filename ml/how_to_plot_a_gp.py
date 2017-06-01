'''
How to plot gaussian processes

Credits:
  https://www.youtube.com/watch?v=4vGiHC35j9s&t=5s
'''

import numpy as np
import matplotlib.pyplot as plt

def kernel(a,b):
  sqdist = np.sum(a**2, 1).reshape(-1, 1)+np.sum(b**2,1) - 2*np.dot(a,b.T)
  return np.exp(-0.5*sqdist)



n = 50
Xtest = np.linspace(-5, 5, n).reshape(-1, 1)
K_ = kernel(Xtest, Xtest)

L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
f_prior = np.dot(L, np.random.normal(size=(n,10)))

plt.plot(Xtest, f_prior)
plt.show()