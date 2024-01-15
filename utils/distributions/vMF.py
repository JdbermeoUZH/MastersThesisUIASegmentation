import numpy as np
from scipy import special


# Helper functions for the von Mises-Fisher distribution

def Ap(p, kappa, exponentially_scaled=True):
    iv = special.ive if exponentially_scaled else special.iv
    return iv(p/2, kappa) / iv(p/2 - 1, kappa)


def Cp(p, kappa):
    num = kappa ** (p/2 - 1)
    denom = (2 * np.pi) ** (p/2) * special.iv(p/2 - 1, kappa)
    return num / denom


def log_Cp(p, kappa):
    return (p/2 - 1) * np.log(kappa) - p/2 * np.log(2*np.pi) - np.log(special.ive(p/2 - 1, kappa)) - np.abs(kappa)


def fp(x, p, mu, kappa):
    return Cp(p, kappa) * np.exp(kappa * np.dot(mu, x))


def log_fp(x, p, mu, kappa):
    return log_Cp(p, kappa) + kappa * np.dot(mu, x)


def kl_divergence(mu_to, kappa_to, mu_fr, kappa_fr):
    """Calculate `KL(to||fr)`, where `mu_to` and `mu_fr` are directions and `kappa_to` and `kappa_fr` are concentration parameters"""

    p = len(mu_to)
    # num = fp(Ap(p, kappa_to, exponentially_scaled=False) * mu_to, p, mu_to, kappa_to)
    # denom = fp(Ap(p, kappa_to) * mu_to, p, mu_fr, kappa_fr)

    # return np.log(num / denom)

    log_num = log_fp(Ap(p, kappa_to) * mu_to, p, mu_to, kappa_to)
    log_denom = log_fp(Ap(p, kappa_to) * mu_to, p, mu_fr, kappa_fr)
    return log_num - log_denom


def kappa_newton_step(kappa, Rbar, p):
    Ap_kappa = Ap(p, kappa)
    return kappa - (Ap_kappa - Rbar) / (1 - Ap_kappa**2 - (p-1)/kappa * Ap_kappa)


def fit_distribution(x, n_iters=0):
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    direction = x.mean(0)
    Rbar = np.linalg.norm(direction)
    direction /= Rbar

    # Approximation of the concentration parameter:
    p = len(direction)
    kappa = Rbar * (p - Rbar**2) / (1 - Rbar**2)

    for _ in range(n_iters):
        print(kappa)
        kappa = kappa_newton_step(kappa, Rbar, p)

    concentration = kappa

    return direction, concentration
