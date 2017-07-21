import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import powerlaw
from detector import Detector
import numdifftools as nd
from pprint import pprint


def matrix_inverse_unfolding(signal, true_energy, detector_response_matrix, num_bins=10):
    sum_signal_per_chamber = np.sum(signal, axis=1)

    # x_vector =
    detector_matrix, detector_matrix_col, detector_matrix_row = detector_response_matrix

    y_vector = np.histogram(sum_signal_per_chamber, bins=detector_matrix.shape[0])
    plt.hist(y_vector, bins=num_bins, label="Y Vector")
    plt.hist(true_energy, bins=np.linspace(min(true_energy), max(true_energy), num_bins), normed=True,
             label="True Energy")
    plt.title("True Energy vs Y_Vector")
    plt.legend(loc='best')
    plt.show()
    # Get the inverse of the detector response matrix
    inv_detector_response_matrix = np.linalg.inv(detector_matrix)

    x_vector_unf = np.dot(inv_detector_response_matrix, y_vector[0])

    # Error propagation
    V_y = np.diag(y_vector[0])
    V_x_est = np.dot(inv_detector_response_matrix, np.dot(V_y, inv_detector_response_matrix.T))
    sigma_x_unf = np.sqrt(np.diag(V_x_est))

    print('x_unf   \t\t= %s' % str(np.round(x_vector_unf, 2)))
    print('simga_x_unf \t\t= %s' % str(np.round(sigma_x_unf, 2)))
    # Need to compare to underlying PDF, which can just be the counts
    # TODO: Change x_vector to the underlying distribution (either from Detector class, or find here)
    # print('(unf - pdf) / sigma_x \t= %s ' % str(np.round((x_vector_unf - x_vector) / sigma_x_unf, 2)))

    plt.hist(x_vector_unf, bins=num_bins, label="Unfolded Energy")
    plt.hist(true_energy, bins=np.linspace(min(true_energy), max(true_energy), num_bins), normed=False,
             label="True Energy", histtype='step')
    plt.title("Number of Particles: " + str(true_energy.shape[0]))
    plt.legend(loc='best')
    plt.show()

    # plt.hist(bin_center, bins=num_bins, weights=x_vector, label='Underlying Distribution', histtype='step')
    # plt.errorbar(bin_center, x_vector_unf, yerr=sigma_x_unf, fmt='.', label='Unfolding')
    # plt.legend(loc='best')
    # plt.xlabel(r'True / Measured Value $x$')
    # plt.ylabel(r'# Events')

    return

    # raise NotImplementedError


energies = 1000.0 * np.random.power(0.70, 5000)
# energies = normal(loc=1000.0, scale=500, size=1000)
below_zero = energies < 0.0
energies[below_zero] = 1.0

detector = Detector(distribution='gaussian',
                    energy_loss='random',
                    make_noise=True,
                    smearing=True,
                    resolution_chamber=1.,
                    noise=0.,
                    plot=False)

detector_test = Detector(distribution='gaussian',
                         energy_loss='random',
                         make_noise=True,
                         smearing=True,
                         resolution_chamber=1.,
                         noise=0.,
                         plot=False)

test_signal, test_true_hits, test_energies, test_detector_matrix = detector_test.simulate(energies)
signal, true_hits, energies, detector_matrix = detector.simulate(energies)
matrix_inverse_unfolding(test_signal, energies, detector_matrix)


def svd_unfolding(signal, true_energy, detector_response_matrix, num_bins=20):
    u, s, v = np.linalg.svd(detector_response_matrix, full_matrices=True)
    print("U:\n" + str(u))
    print("S:\n" + str(s))
    print("V:\n" + str(v))
    raise NotImplementedError


def llh_unfolding(signal, true_energy, detector_response_matrix, num_bins=20):
    # If we need the Hessian, the Numdifftools should give it to us with this

    # Pretty sure should only need the response matrix Hessian, since that gives the curvature of the probabilities
    # that a given energy is in the correct bucket, so using the gradient descent, descend down the probability curvature
    # to get the most likely true distribution based off the measured values.
    # Not sure what log-likliehood does with it, maybe easier to deal the the probabilities?

    hessian_detector = nd.Hessian(detector_response_matrix)

    def LLH(f, data):
        return np.sum(np.log(f * powerlaw.pdf(data) + (1 - f) * powerlaw.pdf(data)))

    raise NotImplementedError
