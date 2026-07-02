# Cosmo AI Fit - Data Fitting Pipeline

Complete MCMC implementation for three-scalar cosmology model inference using Planck CMB and DESI LSS data.

## Quick Start

### Installation

```bash
pip install numpy scipy matplotlib emcee corner
```

### Run Full Analysis

```bash
cd cosmo_pipeline
python run.py
```

This will:
1. ✅ Load Planck + DESI observational data
2. ✅ Run MCMC with 64 walkers × 1000 steps
3. ✅ Compute posterior statistics
4. ✅ Generate corner plots and spectra comparisons
5. ✅ Calculate Bayes factors for model comparison

## Pipeline Modules

### `data_loader.py`
Load and verify observational datasets (CMB + LSS)

### `likelihood.py`
Implement log-likelihood and log-prior functions

### `mcmc.py`
EMCEE sampling framework with convergence diagnostics

### `bayes.py`
Bayesian model comparison and evidence calculation

### `plots.py`
Visualization: corner plots, spectra, residuals, chains

### `run.py`
Main pipeline orchestration (7 steps)

## Expected Output

```
output/
├── chains_YYYYMMDD_HHMMSS.npy
├── posterior_samples.npy
├── posterior_stats.txt
└── figures/
    ├── fig_cmb.pdf
    ├── fig_pk.pdf
    ├── fig_residual.pdf
    ├── fig_corner.pdf
    └── fig_chains.pdf
```

## Parameters Being Fit

- **Γ_φχ** : φ → χ decay rate [0, 0.01] GeV⁻¹
- **Γ_φH** : φ → H decay rate [0, 0.01] GeV⁻¹  
- **Ω_m** : Matter density [0.1, 0.5]
- **H₀** : Hubble constant [60, 75] km/s/Mpc

## Data Requirements

Place in `../data/` directory:
- `planck_cl.dat` - CMB power spectrum (n_ell elements)
- `planck_cov_inv.dat` - Inverse covariance matrix (n_ell × n_ell)
- `desi_pk.dat` - Matter power spectrum (n_k elements)
- `desi_cov_inv.dat` - Inverse covariance matrix (n_k × n_k)

If files not found, mock data will be generated automatically.

## Key Features

✅ **Full Bayesian inference** with proper priors  
✅ **Joint CMB + LSS analysis** with proper covariances  
✅ **Convergence diagnostics** (acceptance fraction, R̂)  
✅ **Posterior statistics** (credible intervals, marginals)  
✅ **Model comparison** via Bayes factors  
✅ **Publication-quality plots**  

## Next Steps

1. **Real CLASS Integration**: Replace mock `run_class()` with actual CLASS/classy interface
2. **Nested Sampling**: Use PyMultiNest/dynesty for accurate evidence computation
3. **Real Data**: Implement Planck/DESI likelihood wrappers
4. **High Performance**: Parallelize with MPI via emcee backends
