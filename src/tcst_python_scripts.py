# tcst_simulation.py
import numpy as np
from scipy.integrate import odeint

# T-CST Simulation

def continuity_eq(rho, t, epsilon, k_perp):
    H = np.sqrt(rho / 3.0)
    t_perp = 0.1 * t
    delta_rho = epsilon * rho * np.sin(k_perp * t_perp) / (t + 1e-10)
    return -3.0 * H * (rho + delta_rho)


def compute_scale_factor(t, rho):
    a = np.zeros_like(t)
    for i in range(1, len(t)):
        a[i] = a[i-1] + 0.5 * (rho[i] + rho[i-1]) * (t[i] - t[i-1])
    return a


def compute_hubble(t, a):
    da_dt = np.gradient(a, t)
    return da_dt / (a + 1e-10)


def run_simulation(rho_0=8.6e-27, epsilon=0.015, k_perp=1e-10, t_max=1e9, N=1000):
    t = np.linspace(1e-10, t_max, N)
    rho_init = rho_0 * (t_max / 1e-10)**2
    rho = odeint(continuity_eq, rho_init, t, args=(epsilon, k_perp)).flatten()
    a = compute_scale_factor(t, rho)
    H = compute_hubble(t, a)
    z = 1.0 / (t / t_max) - 1.0
    return t, z, rho, a, H


# tcst_mcmc.py
import numpy as np
import emcee
from scipy.integrate import odeint
from scipy.interpolate import interp1d

# MCMC fitting

def log_prior(theta):
    rho_0, epsilon, k_perp = theta
    if (1e-28 < rho_0 < 1e-26 and 0.0 < epsilon < 0.1 and 1e-12 < k_perp < 1e-8):
        return 0.0
    return -np.inf


def log_likelihood(theta, t, z_data, H_data, sigma_H):
    rho_0, epsilon, k_perp = theta
    rho_init = rho_0 * (t[-1] / t[0])**2
    rho = odeint(lambda r, t_: -3*np.sqrt(r/3)*(r + epsilon*r*np.sin(k_perp*0.1*t_)/(t_+1e-10)), rho_init, t).flatten()
    a = np.cumsum(0.5*(rho[1:]+rho[:-1])*(t[1:]-t[:-1]))
    a = np.insert(a, 0, 0.0)
    da_dt = np.gradient(a, t)
    H_model = da_dt / (a + 1e-10)
    chi2 = np.sum(((H_model - H_data)/sigma_H)**2)
    return -0.5*chi2


def log_posterior(theta, t, z_data, H_data, sigma_H):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, z_data, H_data, sigma_H)


def run_mcmc_fit(t, z_data, H_data, sigma_H, ndim=3, nwalkers=50, nsteps=1000):
    pos = [np.array([8.6e-27, 0.015, 1e-10]) + 1e-4*np.random.randn(ndim) for _ in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(t, z_data, H_data, sigma_H))
    sampler.run_mcmc(pos, nsteps, progress=True)
    samples = sampler.get_chain(discard=int(nsteps*0.2), thin=15, flat=True)
    best_fit = np.percentile(samples, 50, axis=0)
    return samples, best_fit


# tcst_visualization.py
import matplotlib.pyplot as plt

def plot_hubble(z, H):
    plt.figure(figsize=(8,4))
    plt.plot(z, H, label='Hubble Parameter H(z)')
    plt.xlabel('Redshift z')
    plt.ylabel('H(z) [units]')
    plt.legend()
    plt.show()


def plot_corner(samples):
    import corner
    fig = corner.corner(samples, labels=['rho_0','epsilon','k_perp'])
    plt.show()
