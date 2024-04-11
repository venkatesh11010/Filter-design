import numpy as np
import matplotlib.pyplot as plt

# epsilon = 0.4
epsilon = 0.4

N = 4

beta = ((np.sqrt(1 + epsilon**2) + 1) / epsilon)**(1/N)

r1 = (beta**2 - 1) / (2 * beta)
r2 = (beta**2 + 1) / (2 * beta)

k = np.arange(0, 2*N)
phi_k = np.pi / 2 + (2 * k + 1) * np.pi / (2 * N)

poles = r1 * np.cos(phi_k) + 1j * r2 * np.sin(phi_k)

plt.scatter(np.real(poles), np.imag(poles), marker='x')
plt.xlim(-1, 1)
plt.ylim(-2, 2)
plt.grid()
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Pole-Zero Plot')
plt.grid(True)
plt.show()
