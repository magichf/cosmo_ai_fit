"""Data loading module for observational datasets"""
import numpy as np
import os
from pathlib import Path

def load_data(data_dir="../data"):
    """
    Load all observational data (Planck CMB + DESI LSS)
    
    Returns:
    --------
    dict with keys:
        - planck_cl: CMB power spectrum [n_ell]
        - planck_cov: Planck covariance inverse [n_ell x n_ell]
        - desi_pk: Matter power spectrum [n_k]
        - desi_cov: DESI covariance inverse [n_k x n_k]
    """
    data_files = {
        "planck_cl": "planck_cl.dat",
        "planck_cov": "planck_cov_inv.dat",
        "desi_pk": "desi_pk.dat",
        "desi_cov": "desi_cov_inv.dat"
    }
    
    data = {}
    
    for key, filename in data_files.items():
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                data[key] = np.loadtxt(filepath)
                print(f"✓ Loaded {key}: {data[key].shape}")
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
                data[key] = None
        else:
            print(f"⚠ File not found: {filepath}")
            data[key] = None
    
    # Generate mock data if files not found (for testing)
    if data["planck_cl"] is None:
        print("\nGenerating mock Planck data...")
        n_ell = 100
        data["planck_cl"] = np.random.randn(n_ell) * 100 + 2000
        data["planck_cov"] = np.eye(n_ell) * 0.01
    
    if data["desi_pk"] is None:
        print("Generating mock DESI data...")
        n_k = 50
        data["desi_pk"] = np.random.randn(n_k) * 100 + 1000
        data["desi_cov"] = np.eye(n_k) * 0.01
    
    return data


def check_data_consistency(data):
    """
    Verify data shapes and covariance matrix properties
    """
    print("\nData consistency check:")
    print("-" * 50)
    
    # Check CMB
    n_ell = len(data["planck_cl"])
    cov_shape = data["planck_cov"].shape
    if cov_shape == (n_ell, n_ell):
        print(f"✓ CMB: n_ell = {n_ell}, cov shape = {cov_shape}")
    else:
        print(f"✗ CMB covariance shape mismatch: {cov_shape} vs ({n_ell}, {n_ell})")
    
    # Check LSS
    n_k = len(data["desi_pk"])
    cov_shape = data["desi_cov"].shape
    if cov_shape == (n_k, n_k):
        print(f"✓ LSS: n_k = {n_k}, cov shape = {cov_shape}")
    else:
        print(f"✗ LSS covariance shape mismatch: {cov_shape} vs ({n_k}, {n_k})")
    
    # Check positive definiteness
    evals_cmb = np.linalg.eigvalsh(data["planck_cov"])
    if np.all(evals_cmb > 0):
        print(f"✓ Planck covariance positive definite (λ_min = {evals_cmb.min():.3e})")
    else:
        print(f"✗ Planck covariance not positive definite")
    
    evals_desi = np.linalg.eigvalsh(data["desi_cov"])
    if np.all(evals_desi > 0):
        print(f"✓ DESI covariance positive definite (λ_min = {evals_desi.min():.3e})")
    else:
        print(f"✗ DESI covariance not positive definite")
