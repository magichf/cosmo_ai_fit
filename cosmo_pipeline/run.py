#!/usr/bin/env python
"""Main pipeline: Load data -> Run MCMC -> Analyze results"""

import sys
import os
import numpy as np
from pathlib import Path

# Add pipeline to path
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_data, check_data_consistency
from likelihood import log_probability, run_class
from mcmc import run_mcmc, compute_posterior_stats, save_chains
from bayes import bayes_factor, compute_evidence_from_samples, model_comparison_summary
from plots import (
    plot_cmb, plot_pk, plot_residual, 
    plot_corner_from_chains, plot_chains
)

def main():
    """
    Main analysis pipeline
    """
    print("\n" + "="*70)
    print("THREE-SCALAR COSMOLOGY: MCMC DATA FITTING")
    print("="*70 + "\n")
    
    # ===== STEP 1: Load Data =====
    print("[1/7] Loading observational data...")
    print("-"*70)
    data = load_data(data_dir="../data")
    check_data_consistency(data)
    
    # ===== STEP 2: Setup MCMC =====
    print("\n[2/7] Setting up MCMC sampling...")
    print("-"*70)
    
    # Initial parameters
    initial_theta = {
        "Gamma_phi_chi": 0.001,
        "Gamma_phi_H": 0.002,
        "Omega_m": 0.3,
        "H0": 67.4
    }
    
    param_names = list(initial_theta.keys())
    
    # Create log-prob wrapper
    def log_prob_fn(theta_dict):
        return log_probability(theta_dict, data)
    
    print(f"Initial parameters: {initial_theta}")
    print(f"Parameter space: 4D")
    
    # ===== STEP 3: Run MCMC =====
    print("\n[3/7] Running MCMC sampling...")
    print("-"*70)
    
    chains, sampler, param_names = run_mcmc(
        log_prob_fn,
        initial_theta,
        nwalkers=64,
        nsteps=1000,
        param_names=param_names,
        seed=42,
        progress=True
    )
    
    # ===== STEP 4: Posterior Analysis =====
    print("\n[4/7] Computing posterior statistics...")
    print("-"*70)
    
    samples, stats = compute_posterior_stats(chains, param_names, burnin=0.3)
    
    # ===== STEP 5: Save Results =====
    print("\n[5/7] Saving results...")
    print("-"*70)
    
    os.makedirs("output", exist_ok=True)
    
    chain_file = save_chains(chains, param_names, output_dir="output")
    np.save("output/posterior_samples.npy", samples)
    
    # Save statistics
    stats_file = "output/posterior_stats.txt"
    with open(stats_file, 'w') as f:
        for name, stat_dict in stats.items():
            f.write(f"{name}:\n")
            for key, val in stat_dict.items():
                f.write(f"  {key}: {val:.6f}\n")
    print(f"✓ Saved posterior statistics: {stats_file}")
    
    # ===== STEP 6: Model Comparison (Mock) =====
    print("\n[6/7] Bayesian model comparison...")
    print("-"*70)
    
    # Mock evidence values (in real analysis, use nested sampling)
    logZ_model = -1000.5  # Mock three-scalar model evidence
    logZ_lcdm = -1005.2   # Mock ΛCDM evidence
    
    BF = bayes_factor(logZ_model, logZ_lcdm)
    
    # Model comparison summary
    models_dict = {
        "Three-Scalar": {"logZ": logZ_model, "nparams": 8},
        "ΛCDM": {"logZ": logZ_lcdm, "nparams": 6}
    }
    model_comparison_summary(models_dict)
    
    # ===== STEP 7: Plotting =====
    print("\n[7/7] Generating plots...")
    print("-"*70)
    
    os.makedirs("output/figures", exist_ok=True)
    
    # Generate mock spectra for comparison
    print("Generating mock spectra for visualization...")
    
    # ΛCDM spectra
    n_ell = 100
    ell = np.arange(2, n_ell+2)
    Cl_lcdm = 2000 * np.exp(-(ell-2) / 50)
    
    # Model spectra (using mean posterior values)
    mean_theta = {name: stats[name]['mean'] for name in param_names}
    Cl_model, _ = run_class(mean_theta)
    
    # CMB plot
    plot_cmb(
        np.column_stack([ell, Cl_lcdm]),
        np.column_stack([ell, Cl_model]),
        output_file="output/figures/fig_cmb.pdf"
    )
    
    # Matter power spectrum
    n_k = 50
    k = np.logspace(-3, -1, n_k)
    Pk_lcdm = 1000 * (k / 0.01) ** 0.96
    _, Pk_model = run_class(mean_theta)
    
    # LSS plot
    plot_pk(
        np.column_stack([k, Pk_lcdm]),
        np.column_stack([k, Pk_model]),
        output_file="output/figures/fig_pk.pdf"
    )
    
    # Residuals
    plot_residual(
        Cl_lcdm,
        Cl_model,
        output_file="output/figures/fig_residual.pdf"
    )
    
    # Corner plot
    print("\nGenerating corner plot (may take a moment)...")
    plot_corner_from_chains(
        chains,
        param_names,
        output_file="output/figures/fig_corner.pdf",
        burnin=0.3
    )
    
    # Chain plot
    plot_chains(
        chains,
        param_names,
        output_file="output/figures/fig_chains.pdf"
    )
    
    # ===== Summary =====
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nOutput files:")
    print(f"  Chains: output/chains_*.npy")
    print(f"  Posterior samples: output/posterior_samples.npy")
    print(f"  Statistics: output/posterior_stats.txt")
    print(f"\nFigures:")
    print(f"  output/figures/fig_cmb.pdf")
    print(f"  output/figures/fig_pk.pdf")
    print(f"  output/figures/fig_residual.pdf")
    print(f"  output/figures/fig_corner.pdf")
    print(f"  output/figures/fig_chains.pdf")
    print()

if __name__ == "__main__":
    main()
