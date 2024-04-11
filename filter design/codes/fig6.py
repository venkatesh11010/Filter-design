import numpy as np
import matplotlib.pyplot as plt

# Filter number
L = 1

# Sampling frequency (kHz)
Fs = 48

# Constant used to get the normalized digital frequencies
const = 2 * np.pi / Fs

# The permissible filter amplitude deviation from unity
delta = 0.15

# Bandpass filter specifications (kHz)
F_p1 = 4 + 0.6 * (L + 2)
F_p2 = 4 + 0.6 * (L)

# Transition band
delF = 0.3

# Stopband F_s2 < F_p21; F_p1 < F_s1
F_s1 = F_p1 + 0.3
F_s2 = F_p2 - 0.3

# Normalized digital filter specifications (radians/sec)
omega_p1 = const * F_p1
omega_p2 = const * F_p2

omega_c = (omega_p1 + omega_p2) / 2
omega_l = (omega_p1 - omega_p2) / 2

omega_s1 = const * F_s1
omega_s2 = const * F_s2
delomega = 2 * np.pi * delF / Fs

# The Kaiser window design
A = -20 * np.log10(delta)
N = int(np.ceil((A - 8) / (4.57 * delomega)))
N = 100
n = np.arange(-2 * N, 2 * N + 1)

hlp = np.where(n != 0, np.sin(n * omega_l) / (n * np.pi), omega_l / np.pi) * (abs(n) < N)

# The Bandpass filter
hbp = 2 * hlp * np.cos(n * omega_c)

# The bandpass filter frequency response plot
omega = np.linspace(-np.pi / 2, np.pi / 2, 500)
Hbp = np.abs(np.polyval(hbp, np.exp(-1j * omega)))
plt.plot(omega / np.pi, Hbp)
plt.grid(True)
plt.xlabel('$\\omega$/$\\pi$')
plt.ylabel('|$H_{bp}$($\\omega$)|')
plt.savefig('Hbp_frequency.png')  # Include filename extension in the call
plt.show()
plt.clf()

# The bandpass filter impulse response plot
plt.stem(n, hbp, linefmt='blue', basefmt='r', markerfmt='co')
plt.grid(True)
plt.xlabel('n')
plt.ylabel('|$h_{bp}$(n)|')
plt.savefig('Hbp_impulse.png')  # Include filename extension in the call
plt.show()
