import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.01, 0.99, 500)
y = np.log(x / (1 - x))

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("logit(x)")
plt.title("Logit Function")
plt.grid(True)
plt.savefig("../fig/logit.png")
plt.show()


