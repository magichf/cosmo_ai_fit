#!/usr/bin/env python
"""Quick-start example: minimal working MCMC pipeline

For users who want to get running immediately without full customization.

Usage:
    python quickstart.py
"""

import numpy as np
import emcee
import corner
from pathlib import Path

# ============================================================================
# 1. MOCK THEORY MODEL (three-scalar cosmology)
# ============================================================================

def run_model(theta):
    """
    Compute theoretical CMB and matter power spectra
    
    theta = [Omega_m, H0, Gamma_phi_chi, Gamma_phi_H]
    """
    Omega_m, H0, Gc, Gh = theta
    
    # CMB power spectrum
    ell = np.arange(2, 2500)
    Cl = 1e3 * (ell / 220) ** (-1.1) * np.exp(-(ell / 1200) ** 2)
    
    # Three-scalar decay correction
    decay_strength = 1.0 - 0.1 * (Gc + Gh)
    Cl *= decay_strength
    
    # Matter power spectrum
    k = np.logspace(-3, 0, 200)
    Pk = 1e3 * (k / 0.02) ** 0.96 / (1 + (k / 1.0) ** 2)
    
    # Dark matter generation effect
    Pk *= (1.0 + 0.5 * Gc)
    
    return ell, Cl, k, Pk


# ============================================================================
# 2. MOCK OBSERVATIONAL DATA
# ============================================================================

print("Generating mock observational data...")

# Mock Planck CMB data
ell_data = np.arange(2, 2500)
Cl_obs = 1e3 * (ell_data / 220) ** (-1.1) * np.exp(-(ell_data / 1200) ** 2)
Cl_obs += np.random.randn(len(Cl_obs)) * 0.05 * Cl_obs  # Add noise

# Mock DESI power spectrum data
k_data = np.logspace(-3, 0, 200)
Pk_obs = 1e3 * (k_data / 0.02) ** 0.96 / (1 + (k_data / 1.0) ** 2)
Pk_obs += np.random.randn(len(Pk_obs)) * 0.05 * Pk_obs  # Add noise


# ============================================================================
# 3. LIKELIHOOD FUNCTION
# ============================================================================

def loglike(theta):
    """
    Log-likelihood: chi-squared for CMB + LSS
    """
    try:
        ell, Cl_th, k, Pk_th = run_model(theta)
        
        # Interpolate to data points
        Cl_interp = np.interp(ell_data, ell, Cl_th)
        Pk_interp = np.interp(k_data, k, Pk_th)
        
        # Chi-squared
        sigma_cl = 0.05 * Cl_obs
        sigma_pk = 0.05 * Pk_obs
        
        chi2_cl = np.sum((Cl_interp - Cl_obs) ** 2 / sigma_cl ** 2)
        chi2_pk = np.sum((Pk_interp - Pk_obs) ** 2 / sigma_pk ** 2)
        
        return -0.5 * (chi2_cl + chi2_pk)
    
    except:
        return -np.inf


# ============================================================================
# 4. PRIOR
# ============================================================================

def logprior(theta):
    """
    Uniform priors on all parameters
    """
    Omega_m, H0, Gc, Gh = theta
    
    if (0.1 < Omega_m < 0.5 and 
        60 < H0 < 75 and 
        0 < Gc < 0.01 and 
        0 < Gh < 0.01):
        return 0.0
    
    return -np.inf


def logprob(theta):
    """
    Log-posterior = log-prior + log-likelihood
    """
    lp = logprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(theta)


# ============================================================================
# 5. RUN MCMC
# ============================================================================

print("\n" + "="*70)
print("RUNNING MCMC (emcee)")
print("="*70 + "\n")

ndim = 4
nwalkers = 32
nsteps = 2000

# Initialize walkers
p0 = np.random.uniform(
    low=[0.2, 65, 0.0001, 0.0001],
    high=[0.4, 70, 0.01, 0.01],
    size=(nwalkers, ndim)
)

# Create sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob)

# Run MCMC
print(f"Walkers: {nwalkers}")
print(f"Steps: {nsteps}")
print(f"Total samples: {nwalkers * nsteps}\n")

sampler.run_mcmc(p0, nsteps, progress=True)

# Get posterior samples (after burn-in)
burn_in = 500
samples = sampler.get_chain(discard=burn_in, thin=10, flat=True)

print(f"\n✓ MCMC complete")
print(f"  Posterior samples: {samples.shape[0]}")
print(f"  Mean acceptance: {np.mean(sampler.acceptance_fraction):.3f}\n")


# ============================================================================
# 6. POSTERIOR STATISTICS
# ============================================================================

print("Posterior Statistics:")
print("-" * 70)

param_names = [r"$\Omega_m$", r"$H_0$", r"$\Gamma_{\phi \chi}$", r"$\Gamma_{\phi H}$"]

for i, name in enumerate(param_names):
    vals = samples[:, i]
    q16, q50, q84 = np.percentile(vals, [16, 50, 84])
    print(f"{name:20s}: {q50:.6f} +{q84-q50:.6f} -{q50-q16:.6f}")

print()


# ============================================================================
# 7. BAYES FACTOR
# ============================================================================

log_prob_all = sampler.get_log_prob(flat=True)
logZ_model = np.max(log_prob_all) - np.log(len(log_prob_all))
logZ_lcdm = logZ_model - 2.0  # Mock baseline

bayes_factor = np.exp(logZ_model - logZ_lcdm)

print("="*70)
print(f"Bayes Factor: {bayes_factor:.2f}")
print("="*70)
print()


# ============================================================================
# 8. GENERATE PLOTS
# ============================================================================

import matplotlib.pyplot as plt

print("Generating plots...\n")

# Mean posterior parameters
mean_theta = np.mean(samples, axis=0)
ell, Cl_th, k, Pk_th = run_model(mean_theta)

# Create output directory
output_dir = Path("quickstart_output")
output_dir.mkdir(exist_ok=True)

# --- Plot 1: CMB ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.semilogy(ell_data, Cl_obs, 'o-', label="Data", alpha=0.7, markersize=3)
ax1.semilogy(ell, Cl_th, '--', label="Model", linewidth=2)
ax1.set_ylabel(r"$C_\ell$ [$\mu K^2$]")
ax1.set_title("CMB Power Spectrum")
ax1.legend()
ax1.grid(True, alpha=0.3)

residual_cl = (np.interp(ell_data, ell, Cl_th) - Cl_obs) / Cl_obs
ax2.plot(ell_data, residual_cl, 'o-', markersize=3)
ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
ax2.set_xlabel(r"$\ell$")
ax2.set_ylabel("Residual")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "fig_cmb.pdf", dpi=150)
print(f"  ✓ Saved: fig_cmb.pdf")
plt.close()

# --- Plot 2: Matter Power Spectrum ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.loglog(k_data, Pk_obs, 'o-', label="Data", alpha=0.7, markersize=3)
ax1.loglog(k, Pk_th, '--', label="Model", linewidth=2)
ax1.set_ylabel(r"$P(k)$ [$(Mpc/h)^3$]")
ax1.set_title("Matter Power Spectrum")
ax1.legend()
ax1.grid(True, alpha=0.3, which='both')

residual_pk = (np.interp(k_data, k, Pk_th) - Pk_obs) / Pk_obs
ax2.semilogx(k_data, residual_pk, 'o-', markersize=3)
ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
ax2.set_xlabel(r"$k$ [h/Mpc]")
ax2.set_ylabel("Residual")
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(output_dir / "fig_pk.pdf", dpi=150)
print(f"  ✓ Saved: fig_pk.pdf")
plt.close()

# --- Plot 3: Corner Plot ---
fig = corner.corner(
    samples,
    labels=param_names,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 10},
    smooth=1.0
)
plt.savefig(output_dir / "fig_corner.pdf", dpi=150)
print(f"  ✓ Saved: fig_corner.pdf")
plt.close()

# --- Plot 4: MCMC Chains ---
chains = sampler.get_chain()

fig, axes = plt.subplots(4, 1, figsize=(12, 10))
for i in range(4):
    for j in range(nwalkers):
        axes[i].plot(chains[:, j, i], alpha=0.2, linewidth=0.5)
    axes[i].set_ylabel(param_names[i])
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel("Step")
plt.tight_layout()
plt.savefig(output_dir / "fig_chains.pdf", dpi=150)
print(f"  ✓ Saved: fig_chains.pdf")
plt.close()


# ============================================================================
# 9. SUMMARY
# ============================================================================

print(f"\nOutput directory: {output_dir.absolute()}")
print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print(f"\nFigures generated:")
print(f"  1. fig_cmb.pdf          - CMB power spectrum + residuals")
print(f"  2. fig_pk.pdf           - Matter power + residuals")
print(f"  3. fig_corner.pdf       - Posterior marginal distributions")
print(f"  4. fig_chains.pdf       - MCMC chain traces")
print(f"\nKey results:")
print(f"  • Posterior samples: {samples.shape[0]:,}")
print(f"  • Bayes Factor: {bayes_factor:.2f}")
print(f"  • Mean acceptance: {np.mean(sampler.acceptance_fraction):.1%}")
print()
