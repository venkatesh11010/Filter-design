import numpy as np
import matplotlib.pyplot as plt

# Define the order of the filter
N = 4

# Define the range of epsilon values with desired precision
epsilons = np.round(np.arange(0.35, 0.61, 0.05), 2)  # rounding to 2 decimal places

# Define the frequency range
Omega = np.arange(0, 3.02, 0.02)

# Plot the filter response for each epsilon
# plt.figure()
for epsilon in epsilons:
    H = np.where(Omega < 1, 1 / np.sqrt(1 + epsilon**2 * (np.cos(N * np.arccos(Omega)))**2), 1 / np.sqrt(1 + epsilon**2 * (np.cosh(N * np.arccosh(Omega)))**2))
    plt.plot(Omega, H, label=f'$\\epsilon$ = {epsilon}')

passband = (Omega >= 0) & (Omega <= 1)
stopband = (Omega >= 2) & (Omega <= 3)
transition_band = (Omega >= 1) & (Omega <= 2)
plt.fill_between(Omega, 0, 1, where=passband, color='yellow', alpha=0.5, label='Passband')
plt.fill_between(Omega, 0, 1, where=transition_band, color='magenta', alpha=0.5, label='Transition Band')
plt.fill_between(Omega, 0, 1, where=stopband, color='green', alpha=0.25, label='Stopband')
plt.grid()
plt.xlabel('$\\Omega$')
plt.ylabel('|$H_{a,LP}$(j$\\Omega$)|')
plt.legend()
plt.show()
