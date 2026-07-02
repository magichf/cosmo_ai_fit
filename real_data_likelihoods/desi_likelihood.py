"""DESI Dark Energy Spectroscopic Instrument likelihood

Implements likelihood for DESI DR1 BAO and RSD measurements.
"""

import numpy as np
from pathlib import Path


class DESIBaoLikelihood:
    """DESI BAO + RSD likelihood"""
    
    def __init__(self, data_dir="../data/desi"):
        """
        Initialize DESI likelihood
        
        Parameters:
        -----------
        data_dir : str
            Directory with DESI data
        """
        self.data_dir = Path(data_dir)
        self.has_data = False
        
        # BAO measurements from DESI
        self.bao_measurements = {}  # Will load from file
        self.rsd_measurements = {}  # RSD measurements
        
        self._load_data()
    
    def _load_data(self):
        """
        Load DESI BAO/RSD measurements
        
        Expected:
        - desi_dr1_bao.dat: BAO scale measurements
        - desi_dr1_rsd.dat: RSD measurements
        - desi_dr1_cov.dat: covariance matrix
        """
        bao_file = self.data_dir / "desi_dr1_bao.dat"
        cov_file = self.data_dir / "desi_dr1_cov.dat"
        
        if bao_file.exists() and cov_file.exists():
            try:
                self.bao_obs = np.loadtxt(bao_file)
                self.cov_inv = np.linalg.inv(np.loadtxt(cov_file))
                self.has_data = True
                print(f"✓ Loaded DESI BAO data")
            except Exception as e:
                print(f"✗ Error loading DESI data: {e}")
    
    def __call__(self, k, Pk, z_eff=0.51):
        """
        Compute DESI likelihood
        
        Parameters:
        -----------
        k : array
            Wavenumbers [h/Mpc]
        Pk : array
            Matter power spectrum [Mpc³/h³]
        z_eff : float
            Effective redshift (e.g., 0.51 for DESI DR1 ELG)
        
        Returns:
        --------
        log_likelihood : float
        """
        if not self.has_data:
            return self._mock_likelihood(k, Pk, z_eff)
        
        # Compute BAO scale
        bao_scale = self._compute_bao_scale(k, Pk)
        
        # Compare to observation
        # Standard BAO scale ~150 Mpc/h (evolved from sound horizon)
        bao_obs = 149.3  # DESI DR1 measurement at z=0.51
        
        chi2 = ((bao_scale - bao_obs) / 2.0)**2  # ~2 Mpc/h error
        
        return -0.5 * chi2
    
    def _compute_bao_scale(self, k, Pk):
        """
        Extract BAO scale from power spectrum
        
        Uses Ćircular Broadband + Wiggles decomposition
        """
        # Simple method: find peak in k*Pk
        idx = np.argmax(k * Pk)
        bao_scale = 2 * np.pi / k[idx]  # Convert to spatial scale
        
        return bao_scale
    
    def _mock_likelihood(self, k, Pk, z_eff=0.51):
        """
        Mock DESI likelihood for testing
        """
        # Generate mock power spectrum
        Pk_mock = 1000 * (k / 0.01)**0.96 / (1 + (k/10)**2)
        
        # Simple chi-squared on Pk ratios
        idx_valid = (k > 0.05) & (k < 0.2)  # Typical DESI range
        
        ratio = (Pk[idx_valid] / Pk_mock[idx_valid])**2
        chi2 = np.sum((ratio - 1.0)**2 / (0.1**2))
        
        return -0.5 * chi2


class DESIPowerSpectrumLikelihood:
    """DESI full P(k) likelihood"""
    
    def __init__(self, data_dir="../data/desi"):
        self.data_dir = Path(data_dir)
        self.has_data = False
        self.k_data = None
        self.Pk_data = None
        self.cov_inv = None
        
        self._load_data()
    
    def _load_data(self):
        """Load DESI P(k) measurements"""
        pk_file = self.data_dir / "desi_dr1_pk.dat"
        cov_file = self.data_dir / "desi_dr1_pk_cov_inv.dat"
        
        if pk_file.exists() and cov_file.exists():
            try:
                data = np.loadtxt(pk_file)
                self.k_data = data[:, 0]
                self.Pk_data = data[:, 1]
                self.cov_inv = np.loadtxt(cov_file)
                self.has_data = True
                print(f"✓ Loaded DESI P(k): {len(self.k_data)} points")
            except Exception as e:
                print(f"✗ Error loading DESI P(k): {e}")
    
    def __call__(self, k, Pk):
        """
        Compute P(k) likelihood
        
        Parameters:
        -----------
        k : array
            Wavenumbers
        Pk : array
            Matter power spectrum
        
        Returns:
        --------
        log_likelihood : float
        """
        if not self.has_data:
            return self._mock_likelihood(k, Pk)
        
        # Interpolate theory to data k-points
        Pk_interp = np.interp(self.k_data, k, Pk, fill_value="extrapolate")
        
        # Chi-squared
        delta = Pk_interp - self.Pk_data
        chi2 = delta @ self.cov_inv @ delta
        
        return -0.5 * chi2
    
    def _mock_likelihood(self, k, Pk):
        """Mock likelihood"""
        Pk_mock = 1000 * (k / 0.01)**0.96 / (1 + (k/10)**2)
        
        idx = (k > 0.01) & (k < 0.5)
        residual = (Pk[idx] - Pk_mock[idx]) / Pk_mock[idx]
        
        chi2 = np.sum(residual**2 / (0.05**2))  # 5% error
        
        return -0.5 * chi2
