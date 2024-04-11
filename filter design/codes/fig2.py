import numpy as np
import matplotlib.pyplot as plt

epsilon = 0.4
N = 4

beta = ((np.sqrt(1 + epsilon**2) + 1) / epsilon) ** (1 / N)
r1 = (beta**2 - 1) / (2 * beta)
r2 = (beta**2 + 1) / (2 * beta)

def C_N(x) :
    return 8 * x**4 - 8 * x**2 + 1

def H_aLP(x) :
    return np.sqrt(1 / (1 + epsilon**2 * C_N(x)**2))

# Obtaining the polynomial approximation for the low pass
# Chebyschev filter to obtain a stable filter
u = np.array([1])
for n in range(N // 2):
    phi = np.pi / 2 + (2 * n + 1) * np.pi / (2 * N)
    v = np.array([1, -2 * r1 * np.cos(phi), r1**2 * np.cos(phi)**2 + r2**2 * np.sin(phi)**2])
    p = np.convolve(v, u)
    u = p

# Evaluating the gain of the stable lowpass filter
# The gain has to be 1/sqrt(1+epsilon^2) at Omega = 1
G = np.abs(np.polyval(p, 1j)) / np.sqrt(1 + epsilon**2)

# Plotting the magnitude response of the stable filter
# and comparing with the desired response for the purpose
# of verification
Omega = np.arange(0, 2, 0.01)
H_stable = np.abs(G / np.polyval(p, 1j * Omega))
plt.plot(Omega, H_stable, 'mo', fillstyle='none', label='Design')
plt.plot(Omega, H_aLP(Omega), 'c', label= 'specification')
plt.grid()
plt.xlabel('$\\Omega$')
plt.ylabel('|$H_{a,LP}$(j$\\Omega$)|')
plt.legend()
plt.show()
