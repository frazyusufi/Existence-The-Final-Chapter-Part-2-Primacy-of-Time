import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants (geometric units: G=c=1)
rho_0 = 8.6e-27  # Critical density (kg/m^3, scaled to match cosmology)
epsilon = 0.015    # t_perp perturbation amplitude (tuned for Hubble tension)
k_perp = 1e-10    # t_2/t_3 wavenumber (Planck-scale suppressed)
t_max = 1e9       # t_1 max (s, ~age of universe)
N = 1000          # Time steps

# Continuity equation: d(rho)/dt_1 + 3H(rho + delta_rho) = 0
# H = sqrt(rho/3), delta_rho = epsilon * rho * sin(k*t_perp)/t_1
def continuity_eq(rho, t, epsilon, k_perp):
    H = np.sqrt(rho / 3.0)
    t_perp = 0.1 * t  # Simplified: t_2/t_3 evolve slower than t_1
    delta_rho = epsilon * rho * np.sin(k_perp * t_perp) / (t + 1e-10)
    return -3.0 * H * (rho + delta_rho)

# Compute scale factor: a(t) = int rho(t') dt'
def compute_scale_factor(t, rho):
    a = np.zeros_like(t)
    for i in range(1, len(t)):
        a[i] = a[i-1] + 0.5 * (rho[i] + rho[i-1]) * (t[i] - t[i-1])
    return a

# Hubble parameter: H = da/dt / a
def compute_hubble(t, a):
    da_dt = np.gradient(a, t)
    return da_dt / (a + 1e-10)  # Avoid division by zero

# Metric perturbation (GW-like): h ~ d(rho)/dx, where x = int rho dt
def compute_metric_perturbation(rho, a, t):
    x = compute_scale_factor(t, rho)  # x as cumulative space
    drho_dx = np.gradient(rho, x)
    return 1e-22 * np.abs(drho_dx) / rho_0  # Scaled to LISA sensitivity

# Main simulation
def run_simulation():
    t = np.linspace(1e-10, t_max, N)  # Avoid t=0 singularity
    rho_init = rho_0 * (t_max / 1e-10)**2  # Initial density (matter-like)
    
    # Solve ODE
    rho = odeint(continuity_eq, rho_init, t, args=(epsilon, k_perp)).flatten()
    
    # Compute derived quantities
    a = compute_scale_factor(t, rho)
    H = compute_hubble(t, a)
    
    # Convert t to redshift z (approximate: z ~ 1/t for matter-dominated)
    z = 1.0 / (t / t_max) - 1.0
    H_kms_Mpc = H * 3.086e19 / 3.156e7  # Convert to km/s/Mpc (1 Mpc = 3.086e22 m)
    
    # Metric perturbation for GW signature
    h = compute_metric_perturbation(rho, a, t)
    
    return t, z, rho, a, H_kms_Mpc, h

# Run and store results
t, z, rho, a, H, h = run_simulation()

# Plot results (for visualization, not saved in Pyodide)
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(z, H, label='T-CST H(z)')
plt.axhline(70.5, color='r', linestyle='--', label='H_0 = 70.5 km/s/Mpc')
plt.xscale('log')
plt.xlabel('Redshift z')
plt.ylabel('H(z) [km/s/Mpc]')
plt.legend()
plt.title('T-CST Hubble Parameter')
plt.subplot(2, 1, 2)
plt.plot(z, h, label='Metric Perturbation h')
plt.xscale('log')
plt.xlabel('Redshift z')
plt.ylabel('Strain h')
plt.legend()
plt.title('T-CST GW-like Perturbation')
plt.tight_layout()

# Save data as CSV (in-memory for Pyodide)
np.savetxt('tcst_results.csv', np.vstack((z, H, h)).T, delimiter=',', 
           header='z,H_kms_Mpc,h', comments='')