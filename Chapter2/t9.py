import numpy as np
import matplotlib.pyplot as plt

def step_func(x):
    return np.where(x > 0, 1, 0)

x1 = np.linspace(-5, 5, 500)
y1 = step_func(x1)

plt.plot(x1, y1)
plt.legend(["step"], loc = "best")
plt.yticks(np.arange(0, 1.2, step=0.5))
plt.grid()
plt.show()
plt.savefig("Chapter2/graph/step_func.png")