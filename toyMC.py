import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon
from scipy.stats import chi2
from scipy.optimize import fmin_l_bfgs_b as minimizer
from scipy.special import gammaln
import pandas as pd
import scipy.interpolate as interpolate


def toy_mc(N, x, gamma, acceptance, threshold, **kwargs):
    """
    Generate Toy Data for trying out the different folding and unfolding methods
    :param N: Number of "chambers" that make up the detector
    :param x: Particle Energy
    :param gamma:
    :param acceptance: The acceptance of the detector, between 0 and 1, that depends on the particle energy x
    :param threshold: The energy threshold for the detectors, below this the particle is not detected
    :return: Generated MC data for testing
    """
    input_function = N*x**(-1.*gamma)

    acceptance = 0

    return 0


def inverse_transform_sampling(data, n_bins=40, n_samples=1000):
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = np.random.rand(n_samples)
    return inv_cdf(r)


def create_exponential_decreasing_function(N, x, gamma):
    beta = N/(x**(-1.0*gamma))
    Y = np.random.exponential(beta, 50000)

    H,X1 = np.histogram(Y, normed=True, bins=250)
    dx = X1[1] - X1[0]
    F1 = np.cumsum(H)*dx

    # "Exact" CDF, matches almost perfectly the one above it
    X2 = np.sort(Y)
    F2 = np.array(range(50000))/float(50000)

    plt.plot(X2, F2)

    plt.plot(X1[1:], F1)
    plt.show()

    #plt.show()
    print(Y)

create_exponential_decreasing_function(100, 2400, 2)