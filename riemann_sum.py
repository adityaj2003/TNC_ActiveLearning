import numpy as np
import matplotlib.pyplot as plt

def lw_plus(z, phi):
    return np.log(1 + np.exp(-2*z*np.sin(phi)))

def lw_minus(z, phi):
    return np.log(1 + np.exp(2*z*np.sin(phi)))

def fw_star(z, phi):
    return 2*z

def eA(alpha, step=0.01):
    sum_approx = 0
    for phi in np.arange(0, alpha, step):
        temp_sum = 0
        for z in np.arange(0, 1+step, step):
            temp_sum += 2/np.pi * fw_star(z, phi) * z * step
        sum_approx = sum_approx + temp_sum * step * np.sin(phi)
    return sum_approx

def eC(alpha, step=0.01):
    sum_approx = 0
    for phi in np.arange((np.pi-alpha)/2, (np.pi+alpha)/2, step):
        temp_sum = 0
        for z in np.arange(0, 1+step, step):
            temp_sum += 2/np.pi * fw_star(z, phi) * z * step
        sum_approx = sum_approx + temp_sum * step * np.sin(phi)
    return sum_approx

def eA_simplified(alpha):
    return 4/3*(1-np.cos(alpha))/np.pi

def eC_simplified(alpha):
    return 8/3*np.sin(alpha/2)/np.pi

def compute_difference_hw_hwstar(alpha, eta):
    eA_val = eA(alpha)
    eC_val = eC(alpha)
    difference = (1 - eta) * eA_val - eta * eC_val
    return difference


alphas = np.arange(0.000001, np.pi, 0.01)


etas = np.array([eA(alpha) / (eA(alpha) + eC(alpha)) for alpha in alphas])


# plt.figure(figsize=(10, 6))
# plt.plot(alphas, etas, label=r"$\eta > \frac{eA(\alpha)}{eA(\alpha) + eC(\alpha)}$ using Riemann")
# plt.fill_between(alphas, etas, 0.5, where=(etas <= 0.5), color='gray', alpha=0.3) #alpha is related to transparency level here
# plt.xlim(0, np.pi)
# plt.ylim(0, 0.5)
# plt.xlabel(r"$\alpha$")
# plt.ylabel(r"$\eta$")
# plt.title("(α, η)")
# plt.legend()
# plt.grid(True)
# etas = np.array([(np.sin(alpha/2)/(1+np.sin(alpha/2))) for alpha in alphas])
# plt.plot(alphas, etas, label=r"$\eta > \frac{eA(\alpha)}{eA(\alpha) + eC(\alpha)}$ using simplification")
# plt.fill_between(alphas, etas, 0.5, where=(etas <= 0.5), color='gray', alpha=0.3) #alpha is related to transparency level here
# plt.xlim(0, np.pi)
# plt.ylim(0, 0.5)
# plt.xlabel(r"$\alpha$")
# plt.ylabel(r"$\eta$")
# plt.title("(α, η)")
# plt.legend()
# plt.grid(True)

# plt.show()




etas = np.arange(0.05, 0.55, 0.05)

plt.figure(figsize=(10, 6))

for eta in etas:
    differences = np.array([compute_difference_hw_hwstar(alpha, eta) for alpha in alphas])
    plt.plot(alphas, differences, label=f'η = {eta:.2f}')

plt.xlabel(r'$\alpha$')
plt.ylabel(r'$L(hw) - L(hw^*)$')
plt.title(r'Difference between $L(hw)$ and $L(hw^*)$ for different $\alpha$ with fixed $\eta$')
plt.legend()
plt.grid(True)
plt.show()





alphas = np.linspace(0, np.pi, 10)
etas = np.arange(0.05, 0.55, 0.05)

plt.figure(figsize=(10, 6))

for alpha in alphas:
    differences = np.array([(1 - eta)*eA(alpha) - eta*eC(alpha) for eta in etas])
    plt.plot(etas, differences, label=f'α = {alpha:.2f}')

plt.xlabel(r'$\eta$')
plt.ylabel(r'$L(hw) - L(hw^*)$')
plt.title(r'Difference between $L(hw)$ and $L(hw^*)$ at different $\eta$s with fixed $\alpha$')
plt.legend()
plt.grid(True)
plt.show()