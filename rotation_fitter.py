import numpy as np
import pandas as pd
import emcee
import corner
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from multiprocessing import Pool

# Define the sine model
def model(params, x):
    amplitude, phase = params
    return amplitude * np.sin(x + phase) 

# Define the log-likelihood function including intrinsic scatter
def log_likelihood(params, x, y, y_err):
    amplitude, phase, sigma0 = params
    model_y = model([amplitude, phase], x)
    sigma2 = y_err**2 + sigma0**2
    return -0.5 * np.sum((y - model_y)**2 / sigma2 + np.log(sigma2))

# Define the log-prior function using the priors object
def log_prior(params, priors):
    amplitude, phase, sigma0 = params
    if (priors.lo_amplitude < amplitude < priors.hi_amplitude and
        priors.lo_phase < phase < priors.hi_phase and
        priors.lo_sigma0 < sigma0 < priors.hi_sigma0):
        return 0.0
    return -np.inf

# Define the log-probability function
def log_probability(params, x, y, y_err, priors):
    lp = log_prior(params, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, x, y, y_err)

# Define the priors class
class Priors:
    def __init__(self):
        self.lo_amplitude = 0
        self.hi_amplitude = np.abs(y).max()
        self.lo_phase = 0 
        self.hi_phase = 2 * np.pi
        self.lo_sigma0 = 0
        self.hi_sigma0 = np.abs(y).max()

# Main script
if __name__ == '__main__':
    # Set cluster name
    num = 104

    # Read in all stars
    dft = pd.read_csv(f'{num}.csv')

    # Read in all stars matched with photometry
    dfp = pd.read_csv(f'{num}_MP.csv')
    dfp1 = dfp[dfp['pop'] == 'P1']
    dfp2 = dfp[dfp['pop'] == 'P2']

    # Calculate theta in degrees
    dft['theta'] = (np.degrees(np.arctan2(dft['ra_c'], dft['dec_c'])) + 360) % 360

    # Data for the fit (converting theta back to radians)
    x, y, y_err = np.deg2rad(dft['theta']), dft['RV'] - (-17.45), dft['ERV']

    # Create the priors object
    priors = Priors()

    # Set up the MCMC sampler
    initial = np.array([5.0, 1.0, 0.5])  # amplitude = 5, phase = 1, sigma0 = 0.5
    nwalkers = 32
    ndim = len(initial)
    pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)

    # Run MCMC
    parallel = True

    if parallel:
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, y_err, priors), pool=pool)
            nsteps = 5000
            sampler.run_mcmc(pos, nsteps, progress=True)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, y_err, priors))
        nsteps = 5000
        sampler.run_mcmc(pos, nsteps, progress=True)

    # Discard burn-in samples and flatten the chain
    burn_in = 1000
    samples = sampler.get_chain(discard=burn_in, flat=True)

    # Extract the best-fit parameters
    best_fit_params = np.median(samples, axis=0)
    print("Best-fit parameters:", best_fit_params)

    # Figures
    x_in_deg = True
    if x_in_deg:
        x_fit_degrees = np.linspace(0, 360, 1000)
        x_fit_radians = np.deg2rad(x_fit_degrees)  # Convert to radians for fitting
        y_fit = model(best_fit_params[:-1], x_fit_radians)
        x_deg = np.rad2deg(x)

    # Plot the corner plot
    labels = ["Amplitude", "Phase (rad)", "$\sigma_0$"]
    fig = corner.corner(samples, labels=labels, 
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 12})
    fig.savefig(f'figures/{num}_corner.png', dpi=200, bbox_inches='tight')

    # Calculate residuals
    residuals = y - model(best_fit_params[:-1], x)

    # Create a figure with a gridspec layout
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])

    # Upper panel
    ax_upper = plt.subplot(gs[0, 0])
    ax_upper.errorbar(x_deg, y, yerr=y_err, fmt='.', linewidth=0.5, alpha=0.5, ms=2, label="Data", zorder=0)
    ax_upper.plot(x_fit_degrees, y_fit, label='Model')
    ax_upper.set_xlabel('Position angle  $ \\theta_0$')
    ax_upper.set_ylabel('$\Delta$V$_{LOS}$')
    ax_upper.legend(ncol=3)

    # Lower panel
    ax_lower = plt.subplot(gs[1, 0])
    ax_lower.errorbar(x_deg, residuals, yerr=y_err, fmt='.', linewidth=0.5, alpha=0.5, ms=2, label="Data", zorder=0)
    ax_lower.axhline(0, color='gray', linestyle='--')
    ax_lower.set_xlabel('Position angle $\\theta_0$')
    ax_lower.set_ylabel('Residuals')

    # Histogram of residuals (right of the lower panel)
    ax_hist = plt.subplot(gs[1, 1], sharey=ax_lower)
    ax_hist.hist(residuals, bins=30, density=True, histtype='step', orientation='horizontal')
    _x = np.linspace(-residuals.max(), residuals.max(), 100)
    sigma = best_fit_params[-1]
    ax_hist.plot(1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(_x - 0)**2 / (2 * sigma**2)), _x, color='tab:orange')
    ax_hist.set_xlabel('Density')
    ax_hist.yaxis.set_label_position("right")
    ax_hist.yaxis.tick_right()

    # Print the standard deviation of the residuals
    print('residual std:', np.std(residuals))

    # Adjust layout
    plt.tight_layout()
    fig.savefig(f'figures/{num}_best_fit.png', dpi=200, bbox_inches='tight')
