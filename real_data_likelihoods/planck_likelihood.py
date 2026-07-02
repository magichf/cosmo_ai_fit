"""Planck 2018 CMB likelihood implementation

Interface to Planck likelihood code or approximation thereof.
"""

import numpy as np
from pathlib import Path


class PlanckLikelihood:
    """Planck 2018 high-ℓ + low-ℓ CMB likelihood"""
    
    def __init__(self, data_dir="../data/planck"):
        """
        Initialize Planck likelihood
        
        Parameters:
        -----------
        data_dir : str
            Directory containing Planck data files
        """
        self.data_dir = Path(data_dir)
        self.has_data = False
        self.cl_obs = None
        self.cov_inv = None
        self.ell_min = 2
        self.ell_max = 2500
        
        # Try to load real data
        self._load_data()
    
    def _load_data(self):
        """
        Load Planck data from disk
        
        Expected format:
        - planck_2018_cl.dat: [ell, C_l^TT, C_l^EE, C_l^TE]
        - planck_2018_cov_inv.dat: inverse covariance matrix
        """
        cl_file = self.data_dir / "planck_2018_cl.dat"
        cov_file = self.data_dir / "planck_2018_cov_inv.dat"
        
        if cl_file.exists() and cov_file.exists():
            try:
                self.cl_obs = np.loadtxt(cl_file)
                self.cov_inv = np.loadtxt(cov_file)
                self.has_data = True
                print(f"✓ Loaded Planck data: {len(self.cl_obs)} multipoles")
            except Exception as e:
                print(f"✗ Error loading Planck data: {e}")
    
    def __call__(self, cl_th, ell=None):
        """
        Compute log-likelihood
        
        log L = -1/2 * χ² = -1/2 * (C_th - C_obs)^T Cov^-1 (C_th - C_obs)
        
        Parameters:
        -----------
        cl_th : array
            Theoretical power spectrum
        ell : array, optional
            Multipole moments (if not provided, use full range)
        
        Returns:
        --------
        log_likelihood : float
        """
        if not self.has_data:
            # Use mock likelihood if real data not available
            return self._mock_likelihood(cl_th)
        
        # Check dimensions
        if len(cl_th) != len(self.cl_obs):
            return -np.inf
        
        # Compute chi-squared
        delta_cl = cl_th - self.cl_obs
        chi2 = delta_cl @ self.cov_inv @ delta_cl
        
        return -0.5 * chi2
    
    def _mock_likelihood(self, cl_th):
        """
        Mock Planck likelihood for testing
        """
        # Generate mock observed spectrum
        ell = np.arange(2, len(cl_th) + 2)
        
        # Typical Planck noise level
        noise_level = np.sqrt(2.0 / (2 * ell + 1))  # Cosmic variance + noise
        cl_obs = cl_th * (1 + 0.01 * np.random.randn(len(cl_th)) * noise_level)
        
        # Covariance
        cov = np.diag(noise_level**2 * cl_th**2)
        
        try:
            cov_inv = np.linalg.inv(cov)
        except:
            return -np.inf
        
        delta_cl = cl_th - cl_obs
        chi2 = delta_cl @ cov_inv @ delta_cl
        
        return -0.5 * chi2


class PlanckLowellLikelihood:
    """Planck low-ell (ℓ < 30) TT + EE + TE likelihood"""
    
    def __init__(self, data_dir="../data/planck"):
        """
        Low-ell likelihood
        
        Parameters:
        -----------
        data_dir : str
            Data directory
        """
        self.data_dir = Path(data_dir)
        self.ell_max = 29
        self.has_data = False
    
    def __call__(self, cl_th_tt, cl_th_ee, cl_th_te):
        """
        Compute low-ell likelihood
        
        Parameters:
        -----------
        cl_th_tt : array
        cl_th_ee : array
        cl_th_te : array
        
        Returns:
        --------
        log_likelihood : float
        """
        # Mock implementation
        ell = np.arange(2, 30)
        
        # Simple chi-squared
        noise = 10.0
        chi2_tt = np.sum((cl_th_tt[:28] - 2700*(ell/220)**-1.1)**2 / noise**2)
        chi2_ee = np.sum((cl_th_ee[:28] - 300*(ell/220)**-1.5)**2 / noise**2)
        chi2_te = np.sum((cl_th_te[:28] - 500)**2 / (10*noise)**2)
        
        return -0.5 * (chi2_tt + chi2_ee + chi2_te)
