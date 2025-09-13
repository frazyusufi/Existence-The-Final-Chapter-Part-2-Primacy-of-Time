import numpy as np
import emcee
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d

# Constants (geometric units: G=c=1)
rho_0_fid = 8.6e-27  # Fiducial critical density (kg/m^3)
epsilon_fid = 0.015   # Fiducial t_perp perturbation amplitude
k_perp_fid = 1e-10    # Fiducial t_2/t_3 wavenumber (s^-1)
t_max = 1e9           # t_1 max (s, ~age of universe)
N = 500              # Time steps
c = 3e8              # Speed of light (m/s)
Mpc_to_m = 3.086e22  # Mpc to meters
s_to_Myr = 3.156e13  # Seconds to Myr

# Mock DESI BAO data (z, H, sigma_H in km/s/Mpc)
z_bao = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5])
H_bao = np.array([73.0, 85.0, 110.0, 150.0, 200.0, 250.0])  # Tuned to local H_0
sigma_bao = np.array([2.0, 1.8, 1.5, 1.3, 1.2, 1.1])

# Mock Pantheon+ SNe data (z, mu, sigma_mu)
z_sne = np.logspace(-2, 0.36, 30)  # z=0.01 to 2.3
mu_sne = 25 + 5 * np.log10(1e6 * c / H_bao[0] * z_sne * (1 + 0.5 * z_sne))  # Approx DL
sigma_sne = 0.1 * np.ones_like(z_sne)  # Typical error ~0.1 mag

# T-CST continuity equation: d(rho)/dt_1 + 3H(rho + delta_rho) = 0
def continuity_eq(rho, t, epsilon, k_perp):
    H = np.sqrt(rho / 3.0)
    t_perp = 0.1 * t  # Simplified t_2/t_3 evolution
    delta_rho = epsilon * rho * np.sin(k_perp * t_perp) / (t + 1e-10)
    return -3.0 * H * (rho + delta_rho)

# Compute scale factor: a(t) = int rho(t') dt'
def compute_scale_factor(t, rho):
    a = np.zeros_like(t)
    for i in range(1, len(t)):
        a[i] = a[i-1] + 0.5 * (rho[i] + rho[i-1]) * (t[i] - t[i-1])
    return a

# Hubble parameter: H = da/dt / a, converted to km/s/Mpc
def compute_hubble(t, a, z_data):
    da_dt = np.gradient(a, t)
    H = da_dt / (a + 1e-10) * c / Mpc_to_m * s_to_Myr  # km/s/Mpc
    z = 1.0 / (t / t_max) - 1.0
    H_interp = interp1d(z, H, bounds_error=False, fill_value="extrapolate")
    return H_interp(z_data)

# Distance modulus: mu = 25 + 5*log10(D_L), D_L = (1+z) * int_0^z c/H(z') dz'
def compute_distance_modulus(z, H_func):
    integrand = lambda zp: c / H_func(zp)
    D_L = np.array([quad(integrand, 0, zi)[0] for zi in z])[0]
    return 25 + 5 * np.log10(D_L * (1 + z) / 1e6)  # Mpc to pc

# Log-likelihood for MCMC
def log_likelihood(theta, z_bao, H_bao, sigma_bao, z_sne, mu_sne, sigma_sne):
    rho_0, epsilon, k_perp = theta
    t = np.linspace(1e-10, t_max, N)
    rho_init = rho_0 * (t_max / 1e-10)**2
    rho = odeint(continuity_eq, rho_init, t, args=(epsilon, k_perp)).flatten()
    a = compute_scale_factor(t, rho)
    
    # H(z) for BAO
    H_model = compute_hubble(t, a, z_bao)
    chi2_bao = np.sum(((H_model - H_bao) / sigma_bao)**2)
    
    # mu(z) for SNe
    H_func = interp1d(z_bao, H_model, bounds_error=False, fill_value="extrapolate")
    mu_model = compute_distance_modulus(z_sne, H_func)
    chi2_sne = np.sum(((mu_model - mu_sne) / sigma_sne)**2)
    
    return -0.5 * (chi2_bao + chi2_sne)

# Log-prior: physical constraints
def log_prior(theta):
    rho_0, epsilon, k_perp = theta
    if (1e-28 < rho_0 < 1e-26 and 0.0 < epsilon < 0.1 and 1e-12 < k_perp < 1e-8):
        return 0.0
    return -np.inf

# Log-posterior
def log_posterior(theta, z_bao, H_bao, sigma_bao, z_sne, mu_sne, sigma_sne):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z_bao, H_bao, sigma_bao, z_sne, mu_sne, sigma_sne)

# MCMC setup
ndim, nwalkers = 3, 50
pos = [np.array([rho_0_fid, epsilon_fid, k_perp_fid]) + 1e-4 * np.random.randn(ndim) for _ in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, 
                                args=(z_bao, H_bao, sigma_bao, z_sne, mu_sne, sigma_sne))

# Run MCMC
sampler.run_mcmc(pos, 1000, progress=False)

# Extract samples
samples = sampler.get_chain(discard=200, thin=15, flat=True)
rho_0_mcmc, epsilon_mcmc, k_perp_mcmc = np.percentile(samples, 50, axis=0)
rho_0_err, epsilon_err, k_perp_err = np.std(samples, axis=0)

# Compute best-fit H(z)
t = np.linspace(1e-10, t_max, N)
rho_init = rho_0_mcmc * (t_max / 1e-10)**2
rho = odeint(continuity_eq, rho_init, t, args=(epsilon_mcmc, k_perp_mcmc)).flatten()
a = compute_scale_factor(t, rho)
H_best = compute_hubble(t, a, z_bao)

# Compute chi^2 and evidence (approximate)
ln_Z = sampler.get_log_prob(discard=200, flat=True).max()  # Crude evidence estimate
chi2 = -2 * log_likelihood([rho_0_mcmc, epsilon_mcmc, k_perp_mcmc], 
                           z_bao, H_bao, sigma_bao, z_sne, mu_sne, sigma_sne)

# Store results as CSV (in-memory)
results = np.vstack((z_bao, H_best, np.ones_like(z_bao) * chi2, 
                     np.ones_like(z_bao) * ln_Z)).T
np.savetxt('tcst_fit_results.csv', results, delimiter=',', 
           header='z,H_kms_Mpc,chi2,ln_Z', comments='')

# Plot corner plot (for visualization, not saved)
import corner
fig = corner.corner(samples, labels=['rho_0', 'epsilon', 'k_perp'], 
                    truths=[rho_0_mcmc, epsilon_mcmc, k_perp_mcmc])
plt.figure(figsize=(8, 4))
plt.plot(z_bao, H_bao, 'o', label='DESI BAO (mock)')
plt.plot(z_bao, H_best, '-', label='T-CST Best Fit')
plt.axhline(70.5, color='r', linestyle='--', label='H_0 = 70.5 km/s/Mpc')
plt.xscale('log')
plt.xlabel('Redshift z')
plt.ylabel('H(z) [km/s/Mpc]')
plt.legend()
plt.title(f'T-CST Fit: chi2={chi2:.1f}, H_0={H_best[0]:.1f}')