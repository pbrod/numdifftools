import numdifftools as nd
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 100)
for i in range(0, 10):
    df = nd.Derivative(np.tanh, n=i)
    y = df(x)
    plt.plot(x, y/np.abs(y).max())

plt.axis('off')
plt.axis('tight')
plt.savefig("fun.png")
plt.clf()
