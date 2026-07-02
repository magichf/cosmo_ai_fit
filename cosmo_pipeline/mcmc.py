"""MCMC sampling module using emcee"""
import numpy as np
import emcee
from datetime import datetime
import os

def theta_dict_to_array(theta_dict, param_names):
    """
    Convert parameter dictionary to array
    """
    return np.array([theta_dict[name] for name in param_names])


def theta_array_to_dict(theta_array, param_names):
    """
    Convert parameter array to dictionary
    """
    return {name: val for name, val in zip(param_names, theta_array)}


def run_mcmc(log_prob_fn, initial_theta, nwalkers=64, nsteps=3000, 
             param_names=None, seed=42, progress=True):
    """
    Run MCMC sampling with emcee
    
    Parameters:
    -----------
    log_prob_fn : callable
        Log-probability function: log_prob_fn(theta_dict) -> float
    initial_theta : dict
        Initial parameter values
    nwalkers : int
        Number of MCMC walkers
    nsteps : int
        Number of steps per walker
    param_names : list
        Parameter names (for ordering)
    seed : int
        Random seed
    progress : bool
        Show progress bar
    
    Returns:
    --------
    chains : array [nsteps x nwalkers x ndim]
        MCMC chains
    sampler : emcee.EnsembleSampler
        Sampler object with acceptance_fraction etc.
    param_names : list
        Parameter names
    """
    np.random.seed(seed)
    
    if param_names is None:
        param_names = list(initial_theta.keys())
    
    ndim = len(param_names)
    
    print("="*70)
    print("EMCEE MCMC SAMPLING")
    print("="*70)
    print(f"Walkers: {nwalkers}")
    print(f"Steps per walker: {nsteps}")
    print(f"Total samples: {nwalkers * nsteps}")
    print(f"Parameters: {param_names}")
    print()
    
    # Initialize walker positions
    initial_array = theta_dict_to_array(initial_theta, param_names)
    spread = 0.1
    perturbation = spread * np.random.randn(nwalkers, ndim)
    p0 = initial_array[np.newaxis, :] * (1 + perturbation)
    
    # Wrap log_prob to handle dictionary <-> array conversion
    def log_prob_wrapper(theta_array):
        theta_dict = theta_array_to_dict(theta_array, param_names)
        return log_prob_fn(theta_dict)
    
    # Create sampler
    sampler = emcee.EnsembleSampler(
        nwalkers, 
        ndim, 
        log_prob_wrapper,
        moves=emcee.moves.StretchMove(a=2.0)
    )
    
    # Run MCMC
    print(f"Running MCMC...")
    sampler.run_mcmc(p0, nsteps, progress=progress)
    
    chains = sampler.get_chain()
    
    print(f"\n✓ MCMC complete")
    print(f"  Chain shape: {chains.shape}")
    print(f"  Mean acceptance: {np.mean(sampler.acceptance_fraction):.3f}")
    
    return chains, sampler, param_names


def compute_posterior_stats(chains, param_names, burnin=0.3):
    """
    Compute posterior statistics from chains
    
    Parameters:
    -----------
    chains : array [nsteps x nwalkers x ndim]
    param_names : list
    burnin : float
        Fraction of steps to discard
    
    Returns:
    --------
    samples : array [nsamples x ndim]
        Flattened posterior samples
    stats : dict
        Statistics for each parameter
    """
    # Remove burn-in
    nburnin = int(burnin * len(chains))
    chains_burnin = chains[nburnin:]
    
    # Flatten: [nsteps-nburnin, nwalkers, ndim] -> [nsamples, ndim]
    samples = chains_burnin.reshape(-1, chains_burnin.shape[-1])
    
    print(f"\nPosterior Statistics (burn-in: {burnin*100:.0f}%):")
    print("-" * 70)
    
    stats = {}
    for i, name in enumerate(param_names):
        vals = samples[:, i]
        
        # Percentiles
        q16, q50, q84 = np.percentile(vals, [16, 50, 84])
        mean = np.mean(vals)
        std = np.std(vals)
        
        stats[name] = {
            'mean': mean,
            'std': std,
            'median': q50,
            'lower': q16,
            'upper': q84,
        }
        
        print(f"{name:15s}: {q50:.6f} +{q84-q50:.6f} -{q50-q16:.6f}")
    
    print()
    return samples, stats


def save_chains(chains, param_names, output_dir="output"):
    """
    Save chains to disk
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save chains
    chain_file = os.path.join(output_dir, f"chains_{timestamp}.npy")
    np.save(chain_file, chains)
    
    # Save parameter names
    meta_file = os.path.join(output_dir, f"chains_{timestamp}_meta.txt")
    with open(meta_file, 'w') as f:
        f.write(f"parameters: {', '.join(param_names)}\n")
        f.write(f"shape: nsteps={chains.shape[0]}, nwalkers={chains.shape[1]}, ndim={chains.shape[2]}\n")
    
    print(f"✓ Saved chains: {chain_file}")
    return chain_file
