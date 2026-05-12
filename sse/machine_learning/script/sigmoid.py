

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 500)
y = 1 / (1 + np.exp(-x))

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("sigmoid(x)")
plt.title("Sigmoid Function")
plt.grid(True)
plt.savefig("../fig/sigmoid.png")
plt.show()
