import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import scipy.stats as stats

# Set random seed for reproducibility
np.random.seed(42)

# Define some constants and simulation parameters
N_GALAXIES = 1000  # Number of galaxies to simulate
N_SAMPLES = 1000   # Number of posterior samples
TUNE_SAMPLES = 1000  # Number of tuning steps for MCMC

def create_gw_galaxy_model(observed_gal_mass=None, observed_gal_distance=None,
                          observed_em=None, observed_gw=None):
    """
    PyMC implementation of the hierarchical Bayesian model for gravitational wave sources
    with galaxy distribution priors.

    Parameters:
    -----------
    observed_gal_mass : array-like, optional
        Observed galaxy masses (in solar masses)
    observed_gal_distance : array-like, optional
        Observed galaxy distances (in Mpc)
    observed_em : array-like, optional
        Binary indicator of electromagnetic detection (1 if detected, 0 otherwise)
    observed_gw : array-like, optional
        Binary indicator of gravitational wave detection (1 if detected, 0 otherwise)

    Returns:
    --------
    model : pm.Model
        PyMC model object
    """
    with pm.Model() as model:
        # ========== HYPERPRIORS FOR GALAXY DISTRIBUTION ==========
        # Average cosmic density of galaxies
        rho_0_mu = np.log(0.1)  # log of 0.1 galaxies/Mpc^3
        rho_0_sigma = 0.3
        rho_0 = pm.LogNormal("rho_0", mu=rho_0_mu, sigma=rho_0_sigma)

        # Mass-dependent bias factor parameters
        alpha = pm.Normal("alpha", mu=0.25, sigma=0.05)
        log_m1 = pm.Normal("log_m1", mu=10, sigma=0.5)
        m1 = pm.Deterministic("m1", 10**log_m1)

        # Large-scale structure parameters
        d0 = pm.Normal("d0", mu=5.0, sigma=0.5)  # Mpc
        gamma = pm.Normal("gamma", mu=1.8, sigma=0.1)

        # Schechter function parameter
        log_m_star = pm.Normal("log_m_star", mu=11, sigma=0.3)
        m_star = pm.Deterministic("m_star", 10**log_m_star)

        # ========== HYPERPRIORS FOR EM OBSERVATIONS ==========
        # Mass-to-luminosity relation parameters
        log_L0 = pm.Normal("log_L0", mu=10, sigma=0.5)
        L0 = pm.Deterministic("L0", 10**log_L0)
        beta = pm.Normal("beta", mu=1.2, sigma=0.1)

        # Telescope detection parameters
        log_f_det = pm.Normal("log_f_det", mu=-8, sigma=0.5)  # Flux detection limit
        f_det = pm.Deterministic("f_det", 10**log_f_det)
        sigma_L = pm.Uniform("sigma_L", lower=0.1, upper=0.5)

        # ========== HYPERPRIORS FOR GW OBSERVATIONS ==========
        # Merger rate density parameters
        log_rho0_gw = pm.LogNormal("log_rho0_gw", mu=np.log(30), sigma=0.7)  # log of 30 Gpc^-3 yr^-1
        rho0_gw = pm.Deterministic("rho0_gw", log_rho0_gw)
        alpha_gw = pm.Normal("alpha_gw", mu=1.0, sigma=0.3)

        # GW detection parameters
        log_d0_gw = pm.LogNormal("log_d0_gw", mu=np.log(100), sigma=0.3)  # log of 100 Mpc
        d0_gw = pm.Deterministic("d0_gw", log_d0_gw)

        # Chirp mass distribution parameters
        w_bns = pm.Beta("w_bns", alpha=2, beta=5)  # Weight for BNS vs BBH

        # ========== GENERATE OR CONDITION ON GALAXY PARAMETERS ==========
        if observed_gal_mass is None or observed_gal_distance is None:
            # If no observations provided, generate galaxies from prior
            # This is a simplified version - actual sampling from galaxy distribution
            # would be more complex

            # Sample galaxy masses from Schechter-like function
            log_mass = pm.Normal("log_mass", mu=10.5, sigma=0.5, shape=N_GALAXIES)
            mass = pm.Deterministic("mass", 10**log_mass)

            # Sample distances with correlation structure (simplified)
            log_distance = pm.Normal("log_distance", mu=np.log(100), sigma=0.7, shape=N_GALAXIES)
            distance = pm.Deterministic("distance", 10**log_distance)
        else:
            # Use observed values
            mass = pm.MutableData("mass", observed_gal_mass)
            distance = pm.MutableData("distance", observed_gal_distance)

        # ========== CALCULATE GALAXY DISTRIBUTION PRIOR ==========
        # Mass-dependent bias factor
        bias = pm.Deterministic("bias", (mass / m1)**alpha)

        # Two-point correlation function (simplified)
        xi = pm.Deterministic("xi", (d0 / distance)**gamma)
        f_d = pm.Deterministic("f_d", 1 + xi)

        # Combined galaxy distribution prior
        galaxy_density = pm.Deterministic(
            "galaxy_density",
            rho_0 * bias * f_d * pm.math.exp(-mass / m_star)
        )

        # ========== ELECTROMAGNETIC OBSERVATION LIKELIHOOD ==========
        # Galaxy luminosity from mass
        luminosity = pm.Deterministic("luminosity", L0 * (mass / 1e11)**beta)

        # Minimum detectable luminosity at given distance
        l_min = pm.Deterministic("l_min", f_det * 4 * np.pi * distance**2)

        # Probability of EM detection
        z_score_em = pm.Deterministic("z_score_em", (pm.math.log(luminosity) - pm.math.log(l_min)) / sigma_L)
        p_em_detect = pm.Deterministic("p_em_detect", pm.math.normcdf(z_score_em))

        # EM detection as Bernoulli process
        if observed_em is None:
            em_detection = pm.Bernoulli("em_detection", p=p_em_detect, shape=len(mass))
        else:
            em_detection = pm.Bernoulli("em_detection", p=p_em_detect, observed=observed_em)

        # ========== GRAVITATIONAL WAVE OBSERVATION LIKELIHOOD ==========
        # Merger rate density scaled by galaxy mass
        merger_rate = pm.Deterministic("merger_rate", rho0_gw * (mass / 1e10)**alpha_gw)

        # Simplified chirp mass distribution (just for illustration)
        # In practice, you'd want to properly sample from a mixture model
        is_bns = pm.Bernoulli("is_bns", p=w_bns, shape=len(mass))
        chirp_mass = pm.Deterministic(
            "chirp_mass",
            is_bns * 1.2 + (1 - is_bns) * 20.0
        )

        # Detection distance threshold based on chirp mass
        d_min = pm.Deterministic("d_min", d0_gw * (chirp_mass / 20.0)**(5/6))

        # Detection probability based on distance
        det_ratio = pm.Deterministic("det_ratio", d_min / distance)
        p_det = pm.Deterministic("p_det", pm.math.minimum(1.0, det_ratio**3))

        # Combined GW detection probability
        # Note: V(m) ∝ m is incorporated into merger_rate scaling
        # T (observation time) is considered a constant factor
        p_gw_detect = pm.Deterministic("p_gw_detect", merger_rate * mass * p_det / 1e3)
        # Rescaling to keep probabilities in a reasonable range

        # GW detection as Bernoulli process
        if observed_gw is None:
            gw_detection = pm.Bernoulli("gw_detection", p=pm.math.minimum(0.99, p_gw_detect), shape=len(mass))
        else:
            gw_detection = pm.Bernoulli(
                "gw_detection",
                p=pm.math.minimum(0.99, p_gw_detect),
                observed=observed_gw
            )

    return model

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
    p_em_detect = stats.norm.cdf(z_score_em)

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

    return {
        'mass': mass,
        'distance': distance,
        'em_detection': em_detection,
        'gw_detection': gw_detection,
        'true_params': true_params
    }

def run_inference(data):
    """
    Run Bayesian inference on the simulated or observed data.

    Parameters:
    -----------
    data : dict
        Dictionary containing observed data

    Returns:
    --------
    trace : arviz.InferenceData
        Inference results
    """
    # Create model with observed data
    model = create_gw_galaxy_model(
        observed_gal_mass=data['mass'],
        observed_gal_distance=data['distance'],
        observed_em=data['em_detection'],
        observed_gw=data['gw_detection']
    )

    # Run inference
    with model:
        # Use NUTS sampler for efficient sampling from high-dimensional distributions
        trace = pm.sample(
            N_SAMPLES,
            tune=TUNE_SAMPLES,
            target_accept=0.9,
            return_inferencedata=True
        )

    return trace

def analyze_results(trace, data):
    """
    Analyze and visualize the inference results.

    Parameters:
    -----------
    trace : arviz.InferenceData
        Inference results
    data : dict
        Dictionary containing simulated data with true parameter values
    """
    # Print summary statistics
    summary = az.summary(trace, var_names=[
        'rho_0', 'alpha', 'm1', 'd0', 'gamma', 'm_star',
        'L0', 'beta', 'f_det', 'sigma_L',
        'rho0_gw', 'alpha_gw', 'd0_gw', 'w_bns'
    ])
    print(summary)

    # Compare to true values
    true_params = data['true_params']
    for param, value in true_params.items():
        if param in summary.index:
            print(f"{param}: True = {value:.4f}, Estimated = {summary.loc[param, 'mean']:.4f}")

    # Plot posterior distributions with true values
    az.plot_forest(
        trace,
        var_names=[
            'rho_0', 'alpha', 'm_star', 'beta',
            'rho0_gw', 'alpha_gw', 'd0_gw', 'w_bns'
        ],
        combined=True,
        figsize=(10, 8)
    )
    plt.tight_layout()
    plt.savefig('posterior_forest.png', dpi=300)

    # Plot trace plots for convergence diagnostics
    az.plot_trace(
        trace,
        var_names=[
            'rho_0', 'alpha', 'm_star', 'beta',
            'rho0_gw', 'alpha_gw', 'd0_gw', 'w_bns'
        ],
        figsize=(12, 10)
    )
    plt.tight_layout()
    plt.savefig('trace_plots.png', dpi=300)

    # Plot posterior predictive checks for EM and GW detections
    with create_gw_galaxy_model(
        observed_gal_mass=data['mass'],
        observed_gal_distance=data['distance']
    ):
        ppc = pm.sample_posterior_predictive(trace, var_names=['em_detection', 'gw_detection'])

    # Check EM detection accuracy
    em_pred_mean = ppc.posterior_predictive.em_detection.mean(dim=("chain", "draw"))
    em_pred = (em_pred_mean > 0.5).values
    em_accuracy = (em_pred == data['em_detection']).mean()
    print(f"EM detection accuracy: {em_accuracy:.4f}")

    # Check GW detection accuracy
    gw_pred_mean = ppc.posterior_predictive.gw_detection.mean(dim=("chain", "draw"))
    gw_pred = (gw_pred_mean > 0.5).values
    gw_accuracy = (gw_pred == data['gw_detection']).mean()
    print(f"GW detection accuracy: {gw_accuracy:.4f}")

def main():
    """
    Main function to run the simulation and inference.
    """
    print("Simulating galaxy data...")
    data = simulate_galaxy_data(N_GALAXIES)

    print("Running Bayesian inference...")
    trace = run_inference(data)

    print("Analyzing results...")
    analyze_results(trace, data)

    print("Done!")

if __name__ == "__main__":
    main()
