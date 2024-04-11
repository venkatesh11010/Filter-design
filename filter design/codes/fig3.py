import numpy as np
import matplotlib.pyplot as plt

def lp_stable_cheb(epsilon, N):
    beta = ((np.sqrt(1 + epsilon**2) + 1) / epsilon) ** (1 / N)
    r1 = (beta**2 - 1) / (2 * beta)
    r2 = (beta**2 + 1) / (2 * beta)

    u = 1
    for n in range(int(N/2)):
        phi = np.pi / 2 + (2 * n + 1) * np.pi / (2 * N)
        v = [1, -2 * r1 * np.cos(phi), r1**2 * np.cos(phi)**2 + r2**2 * np.sin(phi)**2]
        p = np.convolve(v, u)
        u = p

    G = abs(np.polyval(p, 1j)) / np.sqrt(1 + epsilon**2)
    return p, G

def lpbp(p, Omega0, B, Omega_p2):
    N = len(p)
    const = [1, 0, Omega0**2]
    v = const.copy()

    if N > 2:
        for i in range(1, N):
            M = len(v)
            v[M - i - 1] += p[i] * B**i
            if i < N - 1:
                v = np.convolve(const, v)
    elif N == 2:
        M = len(v)
        v[M - 1] += p[N - 1] * B
    den = v

    num = np.concatenate(([1], np.zeros(N - 1)))
    G_bp = abs(np.polyval(den, 1j * Omega_p2) / np.polyval(num, 1j * Omega_p2))

    return num, den, G_bp

# Filter specifications
L = 1
Fs = 48
const = 2 * np.pi / Fs
delta = 0.15
F_p1 = 4 + 0.6 * (L + 2)
F_p2 = 4 + 0.6 * L
F_s1 = F_p1 + 0.3
F_s2 = F_p2 - 0.3
omega_p1 = const * F_p1
omega_p2 = const * F_p2
omega_s1 = const * F_s1
omega_s2 = const * F_s2
Omega_p1 = np.tan(omega_p1 / 2)
Omega_p2 = np.tan(omega_p2 / 2)
Omega_s1 = np.tan(omega_s1 / 2)
Omega_s2 = np.tan(omega_s2 / 2)
Omega_0 = np.sqrt(Omega_p1 * Omega_p2)
B = Omega_p1 - Omega_p2
Omega_Ls = min(abs((Omega_s1**2 - Omega_0**2) / (B * Omega_s1)), abs((Omega_s2**2 - Omega_0**2) / (B * Omega_s2)))

D1 = 1 / ((1 - delta)**2) - 1
D2 = 1 / (delta**2) - 1
N = np.ceil(np.arccosh(np.sqrt(D2 / D1)) / np.arccosh(Omega_Ls))
epsilon1 = np.sqrt(D2) / np.cosh(N * np.arccosh(Omega_Ls))
epsilon2 = np.sqrt(D1)
epsilon = 0.4

# Analog low-pass filter
p, G_lp = lp_stable_cheb(epsilon, N)

# Analog band-pass filter
num, den, G_bp = lpbp(p, Omega_0, B, Omega_p2)

# Plotting
Omega = np.linspace(-0.65, 0.65, 1000)
H_analog_bp = G_bp * np.abs(np.polyval(num, 1j * Omega) / np.polyval(den, 1j * Omega))
plt.plot(Omega, H_analog_bp, 'b')
plt.grid()
plt.xlabel('$\\Omega$')
plt.ylabel('|$H_{a,BP}$(j$\\Omega$)|')
plt.show()
