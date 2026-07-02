"""Wrapper for CLASS (Cosmic Linear Anisotropy Solving System) integration

This module interfaces with classy to compute theoretical CMB and LSS power spectra
for the three-scalar cosmology model.
"""

import numpy as np
from pathlib import Path
import warnings

try:
    from classy import Class
    HAS_CLASSY = True
except ImportError:
    HAS_CLASSY = False
    warnings.warn("classy not installed. Install with: pip install classy-sz")


class ThreeScalarClass:
    """Interface to CLASS for three-scalar cosmology"""
    
    def __init__(self):
        """Initialize CLASS wrapper"""
        self.cosmo = None
        self.is_initialized = False
        self.last_params = None
    
    def set_parameters(self, theta_dict):
        """
        Set cosmological parameters for CLASS
        
        Parameters:
        -----------
        theta_dict : dict
            {
                "Gamma_phi_chi": float,  # φ→χ decay rate
                "Gamma_phi_H": float,    # φ→H decay rate
                "Omega_m": float,        # Matter density
                "H0": float,             # Hubble in km/s/Mpc
                "Omega_b": float,        # Baryon density (optional, default 0.048)
                "n_s": float,            # Scalar tilt (optional, default 0.965)
                "A_s": float,            # Amplitude (optional, default 2.1e-9)
                "tau": float,            # Reionization (optional, default 0.055)
            }
        """
        if not HAS_CLASSY:
            raise ImportError("classy not installed. Install with: pip install classy-sz")
        
        # Extract parameters with defaults
        Omega_m = theta_dict.get("Omega_m", 0.3)
        H0 = theta_dict.get("H0", 67.4)
        Omega_b = theta_dict.get("Omega_b", 0.048)
        n_s = theta_dict.get("n_s", 0.965)
        A_s = theta_dict.get("A_s", 2.1e-9)
        tau = theta_dict.get("tau", 0.055)
        
        Gamma_chi = theta_dict.get("Gamma_phi_chi", 0.001)
        Gamma_H = theta_dict.get("Gamma_phi_H", 0.002)
        
        # Convert H0 to h
        h = H0 / 100.0
        
        # Derived: Omega_Lambda and Omega_cdm
        Omega_cdm = Omega_m - Omega_b
        Omega_Lambda = 1.0 - Omega_m
        
        # CLASS parameters dictionary
        params = {
            "output": "tCl,pCl,mPk",  # Temperature, polarization, matter power
            "h": h,
            "omega_b": Omega_b * h**2,
            "omega_cdm": Omega_cdm * h**2,
            "Omega_Lambda": Omega_Lambda,
            "n_s": n_s,
            "A_s": A_s,
            "tau_reio": tau,
            
            # Precision settings
            "l_max_scalars": 3000,
            "l_max_tensors": 500,
            "P_k_max_1/Mpc": 10.0,
            
            # Three-scalar modifications
            # These would need custom CLASS modifications in production
            # For now, we apply post-processing corrections
        }
        
        self.last_params = {
            "Omega_m": Omega_m,
            "H0": H0,
            "Gamma_chi": Gamma_chi,
            "Gamma_H": Gamma_H,
        }
        
        return params
    
    def compute_spectra(self, theta_dict, k_max=10.0):
        """
        Compute CMB and matter power spectra
        
        Parameters:
        -----------
        theta_dict : dict
            Model parameters
        k_max : float
            Maximum wavenumber in h/Mpc
        
        Returns:
        --------
        ell : array
            Multipole moments
        Cl_TT : array
            CMB temperature power spectrum
        k : array
            Wavenumbers
        Pk : array
            Matter power spectrum
        """
        if not HAS_CLASSY:
            return self._compute_spectra_mock(theta_dict)
        
        try:
            # Get CLASS parameters
            params = self.set_parameters(theta_dict)
            
            # Create or reinitialize CLASS
            if self.cosmo is None:
                self.cosmo = Class()
            
            # Set parameters
            self.cosmo.set(params)
            
            # Compute spectra
            self.cosmo.compute()
            
            # Extract CMB
            ell = np.arange(2, 3001)
            Cl_TT = np.array([self.cosmo.raw_cl(l)["tt"] for l in ell])
            
            # Extract matter power spectrum
            k = np.logspace(-3, np.log10(k_max), 200)
            Pk = np.array([self.cosmo.pk(ki, 0) for ki in k])  # z=0
            
            # Apply three-scalar modifications
            Cl_TT = self._apply_decay_corrections_cmb(Cl_TT, theta_dict)
            Pk = self._apply_decay_corrections_pk(Pk, theta_dict)
            
            self.is_initialized = True
            
            return ell, Cl_TT, k, Pk
        
        except Exception as e:
            print(f"Warning: CLASS computation failed: {e}")
            print("Falling back to mock spectra")
            return self._compute_spectra_mock(theta_dict)
    
    def _apply_decay_corrections_cmb(self, Cl_TT, theta_dict):
        """
        Apply three-scalar decay corrections to CMB
        
        Corrections are modeled as:
        1. Early decay modifies Silk damping
        2. Scale-dependent modification from unequal energy transfer
        """
        Gamma_chi = theta_dict.get("Gamma_phi_chi", 0.001)
        Gamma_H = theta_dict.get("Gamma_phi_H", 0.002)
        
        # Decay strength parameter
        Gamma_tot = Gamma_chi + Gamma_H
        
        # Correction factor: affects high-ell damping
        damping_mod = 1.0 - 0.02 * (Gamma_tot / 0.003)
        
        ell = np.arange(2, len(Cl_TT) + 2)
        
        # Apply scale-dependent damping
        correction = 1.0 + (damping_mod - 1.0) * np.exp(-(ell / 100)**2)
        
        return Cl_TT * correction
    
    def _apply_decay_corrections_pk(self, Pk, theta_dict):
        """
        Apply three-scalar corrections to matter power
        
        Main effect: enhanced growth from dark matter generation
        """
        Gamma_chi = theta_dict.get("Gamma_phi_chi", 0.001)
        Omega_m = theta_dict.get("Omega_m", 0.3)
        
        # Dark matter density modification from decay
        # Rho_chi ∝ Γ_φχ * (matter growth rate)
        growth_enhancement = 1.0 + 0.15 * (Gamma_chi / 0.001) * (Omega_m / 0.3)**0.5
        
        # Matter power grows as D^4 in linear regime
        return Pk * growth_enhancement**2
    
    def _compute_spectra_mock(self, theta_dict):
        """
        Generate mock spectra when classy is not available
        
        This is for testing/demonstration only.
        """
        Omega_m = theta_dict.get("Omega_m", 0.3)
        H0 = theta_dict.get("H0", 67.4)
        Gamma_chi = theta_dict.get("Gamma_phi_chi", 0.001)
        Gamma_H = theta_dict.get("Gamma_phi_H", 0.002)
        
        # Mock CMB
        ell = np.arange(2, 2500)
        Cl_TT = (2700 * (ell / 220)**(-1.1)) / (1 + (ell / 800)**2)
        
        # Apply corrections
        Cl_TT = self._apply_decay_corrections_cmb(Cl_TT, theta_dict)
        
        # Mock matter power
        k = np.logspace(-3, 0, 200)
        Pk = 1000 * (k / 0.01) ** 0.96 / (1 + (k / 10)**2)
        
        # Apply corrections
        Pk = self._apply_decay_corrections_pk(Pk, theta_dict)
        
        return ell, Cl_TT, k, Pk
    
    def get_lcdm_spectra(self, k_max=10.0):
        """
        Get ΛCDM baseline spectra for comparison
        
        Uses Planck 2018 best-fit parameters
        """
        lcdm_params = {
            "Omega_m": 0.315,
            "H0": 67.4,
            "Omega_b": 0.049,
            "n_s": 0.965,
            "A_s": 2.1e-9,
            "tau": 0.055,
            "Gamma_phi_chi": 0.0,  # No decay in ΛCDM
            "Gamma_phi_H": 0.0,
        }
        
        return self.compute_spectra(lcdm_params, k_max)
