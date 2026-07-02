"""Plotting and visualization utilities"""
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False

def plot_cmb(lcdm, model, output_file="fig_cmb.pdf"):
    """
    Compare CMB power spectra: ΛCDM vs Model
    
    Parameters:
    -----------
    lcdm : array [n_ell, 2]
        Columns: ell, Cl_lcdm
    model : array [n_ell, 2]
        Columns: ell, Cl_model
    output_file : str
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Power spectra
    ell_lcdm = lcdm[:, 0] if lcdm.ndim > 1 else np.arange(len(lcdm))
    Cl_lcdm = lcdm[:, 1] if lcdm.ndim > 1 else lcdm
    ell_model = model[:, 0] if model.ndim > 1 else np.arange(len(model))
    Cl_model = model[:, 1] if model.ndim > 1 else model
    
    ax1.plot(ell_lcdm, Cl_lcdm, label="ΛCDM", linewidth=2, color='blue')
    ax1.plot(ell_model, Cl_model, label="Three-Scalar Model", 
             linewidth=2, linestyle='--', color='red')
    ax1.set_ylabel(r"$C_\ell$ [μK²]")
    ax1.set_title("CMB Power Spectrum")
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Residuals
    residual = (Cl_model - Cl_lcdm) / Cl_lcdm
    ax2.plot(ell_model, residual, linewidth=2, color='red')
    ax2.axhline(0, linestyle='--', color='k', alpha=0.5)
    ax2.set_xlabel(r"$\ell$")
    ax2.set_ylabel("Fractional Residual")
    ax2.set_title("Model Deviation from ΛCDM")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()


def plot_pk(lcdm, model, output_file="fig_pk.pdf"):
    """
    Compare matter power spectra: ΛCDM vs Model
    
    Parameters:
    -----------
    lcdm : array [n_k, 2]
        Columns: k, P(k)_lcdm
    model : array [n_k, 2]
        Columns: k, P(k)_model
    output_file : str
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Power spectra
    k_lcdm = lcdm[:, 0] if lcdm.ndim > 1 else np.logspace(-3, 0, len(lcdm))
    Pk_lcdm = lcdm[:, 1] if lcdm.ndim > 1 else lcdm
    k_model = model[:, 0] if model.ndim > 1 else np.logspace(-3, 0, len(model))
    Pk_model = model[:, 1] if model.ndim > 1 else model
    
    ax1.loglog(k_lcdm, Pk_lcdm, label="ΛCDM", linewidth=2, color='blue')
    ax1.loglog(k_model, Pk_model, label="Three-Scalar Model", 
               linewidth=2, linestyle='--', color='red')
    ax1.set_ylabel(r"$P(k)$ [(Mpc/h)³]")
    ax1.set_title("Matter Power Spectrum")
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Residuals
    residual = (Pk_model - Pk_lcdm) / Pk_lcdm
    ax2.semilogx(k_model, residual, linewidth=2, color='red')
    ax2.axhline(0, linestyle='--', color='k', alpha=0.5)
    ax2.set_xlabel(r"$k$ [h/Mpc]")
    ax2.set_ylabel("Fractional Residual")
    ax2.set_title("Model Deviation from ΛCDM")
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()


def plot_residual(lcdm, model, output_file="fig_residual.pdf"):
    """
    Plot residuals
    """
    residual = model - lcdm if model.ndim == 1 else model[:, 1] - lcdm[:, 1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(residual, linewidth=2, color='red')
    plt.axhline(0, linestyle='--', color='k', alpha=0.5)
    plt.xlabel("Index")
    plt.ylabel("Residual")
    plt.title("Model - ΛCDM Residuals")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()


def plot_corner_from_chains(chains, param_names, output_file="fig_corner.pdf", burnin=0.3):
    """
    Create corner plot from MCMC chains
    
    Parameters:
    -----------
    chains : array [nsteps x nwalkers x ndim]
    param_names : list
    output_file : str
    burnin : float
    """
    if not HAS_CORNER:
        print("✗ corner not installed. Skipping corner plot.")
        print("  Install with: pip install corner")
        return
    
    # Remove burn-in and flatten
    nburnin = int(burnin * len(chains))
    chains_burnin = chains[nburnin:]
    samples = chains_burnin.reshape(-1, chains_burnin.shape[-1])
    
    # Create corner plot
    fig = corner.corner(
        samples,
        labels=param_names,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 10},
        smooth=1.0
    )
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()


def plot_chains(chains, param_names, output_file="fig_chains.pdf"):
    """
    Plot parameter chains over iterations
    
    Parameters:
    -----------
    chains : array [nsteps x nwalkers x ndim]
    param_names : list
    output_file : str
    """
    nsteps, nwalkers, ndim = chains.shape
    
    fig, axes = plt.subplots(ndim, 1, figsize=(12, 3*ndim))
    if ndim == 1:
        axes = [axes]
    
    for i in range(ndim):
        for j in range(nwalkers):
            axes[i].plot(chains[:, j, i], alpha=0.3, linewidth=0.5, color='blue')
        axes[i].set_ylabel(param_names[i])
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()
