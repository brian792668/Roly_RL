import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.dirname(os.path.abspath(__file__))

# reward function
x = np.linspace(0, 0.75, 500)
r1 = np.exp(-20 * x**2)
r3 = np.exp(-(50 * x)**4)

# plot
plt.figure(figsize=(7.5, 5))
plt.plot(x, r1, label=r"$r_1 = e^{-20d^2}$", color='blue')
plt.plot(x, r3, label=r"$r_2 = e^{-(50d)^4}$", color='red')

plt.ylim(0, 1)
plt.xlim(0, 0.75)

# plt.title("Plot of $r_1$ and $r_3$")
plt.xlabel("$d$ (m)")
plt.ylabel("$r$ (step reward)")
plt.legend()

plt.grid(alpha=0.3)
plt.savefig(os.path.join(file_path, "Reward function.png"))
plt.show()
