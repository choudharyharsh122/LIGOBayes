import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO
import matplotlib.pyplot as plt
import torch.distributions.transforms as transforms

# Set random seed for reproducibility
pyro.set_rng_seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Constants and simulation parameters
N_GALAXIES = 1000  # Number of galaxies to simulate
N_SAMPLES = 1000   # Number of posterior samples
TUNE_SAMPLES = 1000  # Number of tuning steps for MCMC
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def galaxy_model(observed_gal_mass=None, observed_gal_distance=None,
                observed_em=None, observed_gw=None):
    """
    Pyro implementation of the hierarchical Bayesian model for gravitational wave sources
    with galaxy distribution priors.

    Parameters:
    -----------
    observed_gal_mass : torch.Tensor, optional
        Observed galaxy masses (in solar masses)
    observed_gal_distance : torch.Tensor, optional
        Observed galaxy distances (in Mpc)
    observed_em : torch.Tensor, optional
        Binary indicator of electromagnetic detection (1 if detected, 0 otherwise)
    observed_gw : torch.Tensor, optional
        Binary indicator of gravitational wave detection (1 if detected, 0 otherwise)
    """
    # ========== HYPERPRIORS FOR GALAXY DISTRIBUTION ==========
    # Average cosmic density of galaxies
    rho_0_mu = np.log(0.1)  # log of 0.1 galaxies/Mpc^3
    rho_0_sigma = 0.3
    rho_0 = pyro.sample("rho_0", dist.LogNormal(torch.tensor(rho_0_mu, device=DEVICE),
                                             torch.tensor(rho_0_sigma, device=DEVICE)))

    # Mass-dependent bias factor parameters
    alpha = pyro.sample("alpha", dist.Normal(torch.tensor(0.25, device=DEVICE),
                                         torch.tensor(0.05, device=DEVICE)))
    log_m1 = pyro.sample("log_m1", dist.Normal(torch.tensor(10.0, device=DEVICE),
                                          torch.tensor(0.5, device=DEVICE)))
    m1 = 10**log_m1

    # Large-scale structure parameters
    d0 = pyro.sample("d0", dist.Normal(torch.tensor(5.0, device=DEVICE),
                                    torch.tensor(0.5, device=DEVICE)))  # Mpc
    gamma = pyro.sample("gamma", dist.Normal(torch.tensor(1.8, device=DEVICE),
                                         torch.tensor(0.1, device=DEVICE)))

    # Schechter function parameter
    log_m_star = pyro.sample("log_m_star", dist.Normal(torch.tensor(11.0, device=DEVICE),
                                                  torch.tensor(0.3, device=DEVICE)))
    m_star = 10**log_m_star

    # ========== HYPERPRIORS FOR EM OBSERVATIONS ==========
    # Mass-to-luminosity relation parameters
    log_L0 = pyro.sample("log_L0", dist.Normal(torch.tensor(10.0, device=DEVICE),
                                          torch.tensor(0.5, device=DEVICE)))
    L0 = 10**log_L0
    beta = pyro.sample("beta", dist.Normal(torch.tensor(1.2, device=DEVICE),
                                      torch.tensor(0.1, device=DEVICE)))

    # Telescope detection parameters
    log_f_det = pyro.sample("log_f_det", dist.Normal(torch.tensor(-8.0, device=DEVICE),
                                                torch.tensor(0.5, device=DEVICE)))  # Flux detection limit
    f_det = 10**log_f_det
    sigma_L = pyro.sample("sigma_L", dist.Uniform(torch.tensor(0.1, device=DEVICE),
                                              torch.tensor(0.5, device=DEVICE)))

    # ========== HYPERPRIORS FOR GW OBSERVATIONS ==========
    # Merger rate density parameters
    log_rho0_gw = pyro.sample("log_rho0_gw", dist.LogNormal(torch.tensor(np.log(30), device=DEVICE),
                                                       torch.tensor(0.7, device=DEVICE)))  # log of 30 Gpc^-3 yr^-1
    rho0_gw = log_rho0_gw
    alpha_gw = pyro.sample("alpha_gw", dist.Normal(torch.tensor(1.0, device=DEVICE),
                                             torch.tensor(0.3, device=DEVICE)))

    # GW detection parameters
    log_d0_gw = pyro.sample("log_d0_gw", dist.LogNormal(torch.tensor(np.log(100), device=DEVICE),
                                                   torch.tensor(0.3, device=DEVICE)))  # log of 100 Mpc
    d0_gw = log_d0_gw

    # Chirp mass distribution parameters
    w_bns = pyro.sample("w_bns", dist.Beta(torch.tensor(2.0, device=DEVICE),
                                      torch.tensor(5.0, device=DEVICE)))  # Weight for BNS vs BBH

    # ========== GENERATE OR CONDITION ON GALAXY PARAMETERS ==========
    if observed_gal_mass is None or observed_gal_distance is None:
        # If no observations provided, generate galaxies from prior
        # Sample galaxy masses from Schechter-like function
        log_mass = pyro.sample("log_mass",
                            dist.Normal(torch.tensor(10.5, device=DEVICE),
                                      torch.tensor(0.5, device=DEVICE))
                                     .expand([N_GALAXIES]))
        mass = 10**log_mass

        # Sample distances with correlation structure (simplified)
        log_distance = pyro.sample("log_distance",
                                dist.Normal(torch.tensor(np.log(100), device=DEVICE),
                                          torch.tensor(0.7, device=DEVICE))
                                         .expand([N_GALAXIES]))
        distance = 10**log_distance
    else:
        # Use observed values
        mass = observed_gal_mass
        distance = observed_gal_distance

    # ========== CALCULATE GALAXY DISTRIBUTION PRIOR ==========
    # Mass-dependent bias factor
    bias = (mass / m1)**alpha

    # Two-point correlation function (simplified)
    xi = (d0 / distance)**gamma
    f_d = 1 + xi

    # Combined galaxy distribution prior (not directly used in Pyro, but calculated for completeness)
    galaxy_density = rho_0 * bias * f_d * torch.exp(-mass / m_star)

    # ========== ELECTROMAGNETIC OBSERVATION LIKELIHOOD ==========
    # Galaxy luminosity from mass
    luminosity = L0 * (mass / 1e11)**beta

    # Minimum detectable luminosity at given distance
    l_min = f_det * 4 * np.pi * distance**2

    # Probability of EM detection
    z_score_em = (torch.log(luminosity) - torch.log(l_min)) / sigma_L
    p_em_detect = 0.5 * (1 + torch.erf(z_score_em / torch.sqrt(torch.tensor(2.0, device=DEVICE))))

    # EM detection as Bernoulli process
    if observed_em is None:
        em_detection = pyro.sample("em_detection",
                               dist.Bernoulli(probs=p_em_detect).to_event(1))
    else:
        em_detection = pyro.sample("em_detection",
                               dist.Bernoulli(probs=p_em_detect).to_event(1),
                               obs=observed_em)

    # ========== GRAVITATIONAL WAVE OBSERVATION LIKELIHOOD ==========
    # Merger rate density scaled by galaxy mass
    merger_rate = rho0_gw * (mass / 1e10)**alpha_gw

    # Simplified chirp mass distribution
    is_bns = pyro.sample("is_bns",
                     dist.Bernoulli(probs=w_bns.expand(mass.shape)).to_event(1))
    chirp_mass = is_bns * 1.2 + (1 - is_bns) * 20.0

    # Detection distance threshold based on chirp mass
    d_min = d0_gw * (chirp_mass / 20.0)**(5/6)

    # Detection probability based on distance
    det_ratio = d_min / distance
    p_det = torch.minimum(torch.tensor(1.0, device=DEVICE), det_ratio**3)

    # Combined GW detection probability
    p_gw_detect = merger_rate * mass * p_det / 1e3
    # Rescaling and clamping to keep probabilities in a reasonable range
    p_gw_detect = torch.minimum(torch.tensor(0.99, device=DEVICE), p_gw_detect)
    p_gw_detect = torch.maximum(torch.tensor(1e-10, device=DEVICE), p_gw_detect)

    # GW detection as Bernoulli process
    if observed_gw is None:
        gw_detection = pyro.sample("gw_detection",
                               dist.Bernoulli(probs=p_gw_detect).to_event(1))
    else:
        gw_detection = pyro.sample("gw_detection",
                               dist.Bernoulli(probs=p_gw_detect).to_event(1),
                               obs=observed_gw)

    return {
        "mass": mass,
        "distance": distance,
        "em_detection": em_detection,
        "gw_detection": gw_detection,
        "p_em_detect": p_em_detect,
        "p_gw_detect": p_gw_detect,
        "galaxy_density": galaxy_density,
        "chirp_mass": chirp_mass
    }

def simulate_galaxy_data(n_galaxies=1000):
    """
    Simulate galaxy data according to the model specifications.

    Parameters:
    -----------
    n_galaxies : int
        Number of galaxies to simulate

    Returns:
    --------
    data : dict
        Dictionary containing simulated data
    """
    # Set true parameter values
    true_params = {
        'rho_0': 0.1,          # galaxies/Mpc^3
        'alpha': 0.25,         # bias power law
        'm1': 10**10,          # reference mass
        'd0': 5.0,             # correlation length (Mpc)
        'gamma': 1.8,          # correlation power law
        'm_star': 10**11,      # Schechter cutoff (solar masses)
        'L0': 10**10,          # reference luminosity
        'beta': 1.2,           # mass-to-light power law
        'f_det': 10**(-8),     # flux detection limit
        'sigma_L': 0.25,       # scatter in M/L ratio
        'rho0_gw': 30,         # merger rate (Gpc^-3 yr^-1)
        'alpha_gw': 1.0,       # merger rate mass dependence
        'd0_gw': 100,          # GW detection distance (Mpc)
        'w_bns': 0.3,          # BNS fraction
    }

    # Generate galaxy masses from Schechter-like function
    log_mass = np.random.normal(10.5, 0.5, size=n_galaxies)
    mass = 10**log_mass

    # Generate distances with correlation structure (simplified)
    log_distance = np.random.normal(np.log(100), 0.7, size=n_galaxies)
    distance = 10**log_distance

    # Calculate EM detection probability
    luminosity = true_params['L0'] * (mass / 1e11)**true_params['beta']
    l_min = true_params['f_det'] * 4 * np.pi * distance**2
    z_score_em = (np.log(luminosity) - np.log(l_min)) / true_params['sigma_L']
    p_em_detect = 0.5 * (1 + np.erf(z_score_em / np.sqrt(2)))

    # Generate EM detection outcomes
    em_detection = np.random.binomial(1, p_em_detect)

    # Calculate GW detection probability
    merger_rate = true_params['rho0_gw'] * (mass / 1e10)**true_params['alpha_gw']
    is_bns = np.random.binomial(1, true_params['w_bns'], size=n_galaxies)
    chirp_mass = is_bns * 1.2 + (1 - is_bns) * 20.0
    d_min = true_params['d0_gw'] * (chirp_mass / 20.0)**(5/6)
    p_det = np.minimum(1.0, (d_min / distance)**3)
    p_gw_detect = np.minimum(0.99, merger_rate * mass * p_det / 1e3)

    # Generate GW detection outcomes
    gw_detection = np.random.binomial(1, p_gw_detect)

    # Convert to PyTorch tensors
    return {
        'mass': torch.tensor(mass, device=DEVICE, dtype=torch.float32),
        'distance': torch.tensor(distance, device=DEVICE, dtype=torch.float32),
        'em_detection': torch.tensor(em_detection, device=DEVICE, dtype=torch.float32),
        'gw_detection': torch.tensor(gw_detection, device=DEVICE, dtype=torch.float32),
        'true_params': true_params
    }

def run_inference_mcmc(data):
    """
    Run Bayesian inference on the simulated or observed data using MCMC.

    Parameters:
    -----------
    data : dict
        Dictionary containing observed data

    Returns:
    --------
    samples : dict
        Dictionary containing posterior samples
    """
    # Clear param store before running inference
    pyro.clear_param_store()

    # Set up the model conditioned on observed data
    conditioned_model = pyro.condition(galaxy_model, data={
        "em_detection": data['em_detection'],
        "gw_detection": data['gw_detection']
    })

    # Define model with fixed observed data
    def conditioned_with_fixed_data():
        return conditioned_model(
            observed_gal_mass=data['mass'],
            observed_gal_distance=data['distance']
        )

    # Set up NUTS kernel
    nuts_kernel = NUTS(conditioned_with_fixed_data)

    # Run MCMC
    mcmc = MCMC(nuts_kernel,
                num_samples=N_SAMPLES,
                warmup_steps=TUNE_SAMPLES)

    mcmc.run()

    # Get posterior samples
    samples = mcmc.get_samples()

    return samples

def run_inference_svi(data, num_iterations=5000):
    """
    Run Bayesian inference using Stochastic Variational Inference (SVI).
    This can be faster than MCMC for large models but gives approximate posteriors.

    Parameters:
    -----------
    data : dict
        Dictionary containing observed data
    num_iterations : int
        Number of SVI iterations

    Returns:
    --------
    guide : pyro.infer.autoguide
        Trained variational guide
    """
    # Clear param store before running inference
    pyro.clear_param_store()

    # Set up the model conditioned on observed data
    def conditioned_model():
        return galaxy_model(
            observed_gal_mass=data['mass'],
            observed_gal_distance=data['distance'],
            observed_em=data['em_detection'],
            observed_gw=data['gw_detection']
        )

    # Set up automatic guide
    guide = AutoDiagonalNormal(conditioned_model)

    # Set up SVI optimizer
    adam = pyro.optim.Adam({"lr": 0.01})
    svi = SVI(conditioned_model, guide, adam, loss=Trace_ELBO())

    # Train
    losses = []
    for i in range(num_iterations):
        loss = svi.step()
        losses.append(loss)
        if i % 500 == 0:
            print(f"Iteration {i}/{num_iterations} - Loss: {loss:.4f}")

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('ELBO Loss')
    plt.title('SVI Training Progress')
    plt.yscale('log')
    plt.savefig('svi_loss.png', dpi=300)

    return guide

def sample_posterior_svi(guide, num_samples=1000):
    """
    Sample from the fitted variational posterior.

    Parameters:
    -----------
    guide : pyro.infer.autoguide
        Trained variational guide
    num_samples : int
        Number of samples to draw

    Returns:
    --------
    samples : dict
        Dictionary containing posterior samples
    """
    return guide.sample_posterior(num_samples)

def analyze_results(samples, data, method="MCMC"):
    """
    Analyze and visualize the inference results.

    Parameters:
    -----------
    samples : dict
        Dictionary containing posterior samples
    data : dict
        Dictionary containing simulated data with true parameter values
    method : str
        Inference method used ("MCMC" or "SVI")
    """
    # List of parameters to analyze
    params = ['rho_0', 'alpha', 'log_m1', 'd0', 'gamma', 'log_m_star',
              'log_L0', 'beta', 'log_f_det', 'sigma_L',
              'log_rho0_gw', 'alpha_gw', 'log_d0_gw', 'w_bns']

    # Get true parameter values
    true_params = data['true_params']

    # Create a figure for posterior distributions
    n_rows = int(np.ceil(len(params) / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 3*n_rows))
    axes = axes.flatten()

    # For each parameter, plot posterior and compare to true value
    for i, param in enumerate(params):
        # Get samples for this parameter
        param_samples = samples[param].cpu().numpy()

        # Convert log parameters if needed
        if param.startswith('log_'):
            base_param = param[4:]  # Remove 'log_' prefix
            if base_param in true_params:
                true_val = np.log10(true_params[base_param])
            else:
                true_val = None
        else:
            true_val = true_params.get(param, None)

        # Plot histogram of posterior
        axes[i].hist(param_samples, bins=30, alpha=0.7, density=True)

        # Add true value line if available
        if true_val is not None:
            axes[i].axvline(true_val, color='r', linestyle='--',
                            label=f'True: {true_val:.3f}')

        # Calculate posterior mean and 95% credible interval
        mean_val = np.mean(param_samples)
        ci_lower = np.percentile(param_samples, 2.5)
        ci_upper = np.percentile(param_samples, 97.5)

        # Add posterior mean line
        axes[i].axvline(mean_val, color='g', linestyle='-',
                       label=f'Mean: {mean_val:.3f}')

        # Print comparison if true value is available
        if true_val is not None:
            print(f"{param}: True = {true_val:.4f}, Estimated = {mean_val:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        else:
            print(f"{param}: Estimated = {mean_val:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

        # Add parameter name and legend
        axes[i].set_xlabel(param)
        axes[i].set_ylabel('Density')
        axes[i].legend()

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(f'posterior_distributions_{method}.png', dpi=300)

    # Run posterior predictive check
    predictive = Predictive(galaxy_model, samples)
    predictions = predictive(
        observed_gal_mass=data['mass'],
        observed_gal_distance=data['distance']
    )

    # Check EM detection accuracy
    em_pred_mean = predictions['em_detection'].mean(dim=0).cpu().numpy()
    em_pred = (em_pred_mean > 0.5).astype(np.float32)
    em_actual = data['em_detection'].cpu().numpy()
    em_accuracy = (em_pred == em_actual).mean()
    print(f"EM detection accuracy: {em_accuracy:.4f}")

    # Check GW detection accuracy
    gw_pred_mean = predictions['gw_detection'].mean(dim=0).cpu().numpy()
    gw_pred = (gw_pred_mean > 0.5).astype(np.float32)
    gw_actual = data['gw_detection'].cpu().numpy()
    gw_accuracy = (gw_pred == gw_actual).mean()
    print(f"GW detection accuracy: {gw_accuracy:.4f}")

    # Plot predicted vs actual probabilities
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # EM detection probabilities
    axes[0].scatter(predictions['p_em_detect'].mean(dim=0).cpu().numpy(),
                  em_actual, alpha=0.3)
    axes[0].plot([0, 1], [0, 1], 'r--')
    axes[0].set_xlabel('Predicted EM Detection Probability')
    axes[0].set_ylabel('Actual EM Detection')
    axes[0].set_title('EM Detection Calibration')

    # GW detection probabilities
    axes[1].scatter(predictions['p_gw_detect'].mean(dim=0).cpu().numpy(),
                  gw_actual, alpha=0.3)
    axes[1].plot([0, 1], [0, 1], 'r--')
    axes[1].set_xlabel('Predicted GW Detection Probability')
    axes[1].set_ylabel('Actual GW Detection')
    axes[1].set_title('GW Detection Calibration')

    plt.tight_layout()
    plt.savefig(f'calibration_plots_{method}.png', dpi=300)

def main():
    """
    Main function to run the simulation and inference.
    """
    print("Simulating galaxy data...")
    data = simulate_galaxy_data(N_GALAXIES)

    print(f"Running Bayesian inference with MCMC...")
    try:
        samples = run_inference_mcmc(data)
        print("Analyzing MCMC results...")
        analyze_results(samples, data, method="MCMC")
    except Exception as e:
        print(f"MCMC inference failed: {e}")
        print("Switching to SVI inference...")

    print("Running Bayesian inference with SVI...")
    try:
        guide = run_inference_svi(data)
        samples = sample_posterior_svi(guide)
        print("Analyzing SVI results...")
        analyze_results(samples, data, method="SVI")
    except Exception as e:
        print(f"SVI inference failed: {e}")

    print("Done!")

if __name__ == "__main__":
    main()
