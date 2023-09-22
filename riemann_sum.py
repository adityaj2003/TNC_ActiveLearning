import numpy as np
import matplotlib.pyplot as plt

def lw_plus(z, phi):
    return np.log(1 + np.exp(-2*z*np.sin(phi)))

def lw_minus(z, phi):
    return np.log(1 + np.exp(2*z*np.sin(phi)))

def fw_star(z, phi):
    return lw_minus(z, phi) - lw_plus(z, phi)

def eA(alpha, step=0.01):
    sum_approx = 0
    for phi in np.arange(0, alpha, step):
        for z in np.arange(0, 1+step, step):
            sum_approx += 2/np.pi * fw_star(z, phi) * z * step * step
    return sum_approx

def eC(alpha, step=0.01):
    sum_approx = 0
    for phi in np.arange((np.pi-alpha)/2, np.pi, step):
        for z in np.arange(0, 1+step, step):
            sum_approx += 4/np.pi * fw_star(z, phi) * z * step * step
    return sum_approx

alphas = np.arange(0, np.pi, 0.01)
etas = np.array([eA(alpha) / (eA(alpha) + eC(alpha)) for alpha in alphas])


plt.figure(figsize=(10, 6))
plt.plot(alphas, etas, label=r"$\eta > \frac{eA(\alpha)}{eA(\alpha) + eC(\alpha)}$")
plt.fill_between(alphas, etas, 0.5, where=(etas <= 0.5), color='gray', alpha=0.3) #alpha is related to transparency level here
plt.xlim(0, np.pi)
plt.ylim(0, 0.5)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\eta$")
plt.title("(α, η)")
plt.legend()
plt.grid(True)
plt.show()