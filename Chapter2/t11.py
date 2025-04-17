import numpy as np
import matplotlib.pyplot as plt


def step_func(x):
    return np.where(x > 0, 1, 0)

x1 = np.linspace(-5, 5, 500)
y1 = step_func(x1)

def sigmoid_func(x):
    return 1 / (1 + np.exp(-x))

x2 = np.linspace(-5, 5)
y2 = sigmoid_func(x2)

def tanh_func(x):
    return np.tanh(x)
x3 = np.linspace(-5, 5)
y3 = tanh_func(x3)


plt.plot(x1, y1)
plt.plot(x2, y2)
plt.plot(x3, y3)
plt.legend(["step", "sigmoid", "tanh"], loc = "best")
plt.yticks(np.arange(-1, 1.2, step=0.5))
plt.grid()
plt.show()
plt.savefig("Chapter2/graph/step&sigmoid&tanh_func.png")