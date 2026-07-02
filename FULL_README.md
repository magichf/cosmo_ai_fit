# Three-Scalar Cosmology MCMC Analysis

## 🚀 Quick Start (30 seconds)

```bash
python quickstart.py
```

This runs a complete analysis with mock data and generates 4 publication-quality plots.

## 📊 Full Production Pipeline

### Installation

```bash
# Required packages
pip install numpy scipy matplotlib emcee corner dynesty

# Optional: CLASS integration (recommended)
pip install classy-sz

# Optional: for real Planck likelihoods
pip install clik
```

### Basic Usage

```bash
# Run with mock data
python full_pipeline.py

# Run with real Planck/DESI data
python full_pipeline.py --real-data

# Use nested sampling for evidence
python full_pipeline.py --nested

# Customize MCMC settings
python full_pipeline.py --nsteps 5000 --nwalkers 128
```

## 📁 Project Structure

```
cosmo_ai_fit/
├── full_pipeline.py                 # Production pipeline (recommended)
├── quickstart.py                    # Quick demo with mock data
│
├── cosmo_pipeline/                  # Core analysis modules
│   ├── data_loader.py              # Load Planck/DESI data
│   ├── likelihood.py               # Likelihood functions
│   ├── mcmc.py                     # emcee sampling framework
│   ├── bayes.py                    # Bayesian model comparison
│   └── plots.py                    # Visualization utilities
│
├── class_interface/                 # CLASS integration
│   └── classy_wrapper.py           # Three-scalar CLASS modifications
│
├── real_data_likelihoods/          # Official likelihoods
│   ├── planck_likelihood.py        # Planck 2018 CMB
│   └── desi_likelihood.py          # DESI BAO/RSD/P(k)
│
├── nested_sampling/                 # Evidence calculation
│   └── evidence_calculator.py       # Multiple methods for Z
│
└── data/                            # Observational data
    ├── planck/
    │   ├── planck_2018_cl.dat
    │   └── planck_2018_cov_inv.dat
    └── desi/
        ├── desi_dr1_pk.dat
        └── desi_dr1_cov_inv.dat
```

## 🔬 Pipeline Overview

### Full Pipeline (8 steps)

```python
pipeline = FullAnalysisPipeline(use_real_data=True, use_nested_sampling=True)
pipeline.run_full_pipeline(nsteps=5000, nwalkers=128)
```

**Steps:**
1. Load Planck CMB + DESI LSS data
2. Setup likelihood functions
3. Initialize CLASS interface
4. Run MCMC (64 walkers × 1000 steps default)
5. Compute posterior statistics
6. Calculate Bayesian evidence for model comparison
7. Save chains and statistics
8. Generate publication-quality plots

### Output Structure

```
analysis_output/YYYYMMDD_HHMMSS/
├── chains/
│   ├── chains_YYYYMMDD_HHMMSS.npy
│   └── chains_YYYYMMDD_HHMMSS_meta.txt
├── posterior_samples.npy            # [nsamples × 4] array
├── posterior_stats.txt              # Credible intervals
└── figures/
    ├── fig_cmb.pdf                 # CMB comparison
    ├── fig_pk.pdf                  # Power spectrum comparison
    ├── fig_residual.pdf            # Model residuals
    ├── fig_corner.pdf              # Posterior 2D distributions
    └── fig_chains.pdf              # MCMC traces
```

## 📈 Model Parameters

Being fit with uniform priors:

| Parameter | Range | Description |
|-----------|-------|-------------|
| **Ω_m** | [0.1, 0.5] | Matter density |
| **H₀** | [60, 75] | Hubble constant (km/s/Mpc) |
| **Γ_φχ** | [0, 0.01] | φ→χ decay rate (GeV⁻¹) |
| **Γ_φH** | [0, 0.01] | φ→H decay rate (GeV⁻¹) |

## 🧮 Likelihood Functions

### CMB (Planck 2018)
```python
log L_CMB = -1/2 * (C_th - C_obs)ᵀ Cov⁻¹ (C_th - C_obs)
```

### LSS (DESI)
```python
log L_LSS = -1/2 * (P_th(k) - P_obs(k))ᵀ Cov⁻¹ (P_th(k) - P_obs(k))
```

### Joint
```python
log P(θ|data) = log π(θ) + log L_CMB + log L_LSS
```

## 🎯 Theoretical Model

### Three-Scalar EFT

**Background modifications:**
- φ field decay: φ → χ (dark matter) with rate Γ_φχ
- φ field decay: φ → H (radiation) with rate Γ_φH
- Energy transfer modifies Hubble expansion

**Perturbation effects:**
1. Enhanced dark matter growth from decay
2. Modified sound speed in radiation
3. Scale-dependent corrections to transfer functions

## 📊 Model Comparison

### Bayes Factor

```python
B = exp(log Z_model - log Z_ΛCDM)
```

**Interpretation:**
- B > 150: **Very Strong** evidence for model
- 15 < B < 150: **Strong** evidence
- 3 < B < 15: **Moderate** evidence
- 1 < B < 3: **Weak** evidence
- B ≈ 1: **Inconclusive**

### Evidence Calculation Methods

1. **Harmonic mean** (fast, biased high)
   ```python
   Z ≈ <L(θ_i)>⁻¹
   ```

2. **Laplace approximation** (moderate, assumes Gaussian posterior)
   ```python
   log Z ≈ log L_max + k/2 log(2π) + 1/2 log|Cov|
   ```

3. **Nested sampling** (accurate, more expensive)
   ```python
   dynesty with ~1000 live points
   ```

## 🔧 Advanced Usage

### Custom Prior

```python
def my_prior(theta):
    Omega_m, H0, Gc, Gh = theta
    # Gaussian prior on Omega_m
    if abs(Omega_m - 0.3) > 0.1:
        return -np.inf
    # ... rest of priors
    return 0.0
```

### CLASS Integration

```python
from class_interface import ThreeScalarClass

class_model = ThreeScalarClass()
theta = {"Omega_m": 0.3, "H0": 67.4, 
         "Gamma_phi_chi": 0.001, "Gamma_phi_H": 0.002}

ell, Cl, k, Pk = class_model.compute_spectra(theta)
```

### Real Data Likelihoods

```python
from real_data_likelihoods import (
    PlanckLikelihood,
    DESIPowerSpectrumLikelihood
)

planck = PlanckLikelihood(data_dir="data/planck")
desi = DESIPowerSpectrumLikelihood(data_dir="data/desi")

log_L_cmb = planck(Cl_th)
log_L_lss = desi(k, Pk_th)
```

## 📚 References

- **Emcee**: Foreman-Mackey et al. (2013) - https://emcee.readthedocs.io/
- **Dynesty**: Speagle (2020) - https://dynesty.readthedocs.io/
- **Planck 2018**: Planck Collaboration VI (arXiv:1807.06209)
- **DESI**: DESI Collaboration (2024) - https://arxiv.org/abs/2404.03001
- **CLASS**: Blas, Lesgourgues & Tram (2011) - https://github.com/lesgorgues/class_public

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'classy'"
```bash
pip install classy-sz
```

### "Low acceptance fraction" (< 0.1)
- Increase burn-in: `--nsteps 5000`
- Check likelihood for NaNs
- Reduce `spread` parameter in MCMC initialization

### "Data files not found"
- Mock data will be generated automatically
- Provide real Planck/DESI data in `data/` directory

## 📝 Citation

If you use this pipeline, please cite:

```bibtex
@software{cosmo_ai_fit_2024,
  author = {Your Name},
  title = {Three-Scalar Cosmology MCMC Analysis Pipeline},
  year = {2024},
  url = {https://github.com/magichf/cosmo_ai_fit}
}
```

## 📄 License

MIT License - See LICENSE file for details
