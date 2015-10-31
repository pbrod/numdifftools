import numdifftools as nd
import numpy as np
import matplotlib.pyplot as plt

def fun(x):
    y = np.exp(-x)
    return (1.0 - y)  / ( 1.0 + y)

x = np.linspace(-7, 7, 200)
for i in range(0,7):
    df = nd.Derivative(fun, n=i)
    plt.plot(x, df(x))

plt.axis('off')
plt.savefig("fun.png")
plt.clf()
