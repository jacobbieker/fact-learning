import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import powerlaw
from detector import Detector


def matrix_inverse_unfolding(signal, true_energy, detector_response_matrix, num_bins=20):

    sum_chamber_per_chamber = np.sum(true_energy, axis=0)
    sum_signal_per_chamber = np.sum(signal, axis=0)
    y_vector = sum_signal_per_chamber
    #x_vector =

    detector_matrix, col_norm, row_norm = detector_response_matrix
    print(detector_matrix)
    # Get the inverse of the detector response matrix
    inv_detector_response_matrix = np.linalg.inv(detector_matrix)
    print(inv_detector_response_matrix)

    x_vector_unf = np.dot(inv_detector_response_matrix, y_vector)

    # Error propagation
    V_y = np.diag(y_vector)
    V_x_est = np.dot(inv_detector_response_matrix, np.dot(V_y, inv_detector_response_matrix.T))
    sigma_x_unf = np.sqrt(np.diag(V_x_est))

    print('x_unf   \t\t= %s' % str(np.round(x_vector_unf, 2)))
    print('simga_x_unf \t\t= %s' % str(np.round(sigma_x_unf, 2)))
    # Need to compare to underlying PDF, which can just be the counts
    # TODO: Change x_vector to the underlying distribution (either from Detector class, or find here)
    #print('(unf - pdf) / sigma_x \t= %s ' % str(np.round((x_vector_unf - x_vector) / sigma_x_unf, 2)))

    plt.hist(x_vector_unf, bins=num_bins)
    plt.hist(sum_chamber_per_chamber, bins=np.linspace(min(sum_chamber_per_chamber), max(sum_chamber_per_chamber), 50), normed=False,
             label="True Energy", histtype='step')
    plt.title("Number of Particles: " + str(true_energy.shape[0]))
    plt.show()

    # plt.hist(bin_center, bins=num_bins, weights=x_vector, label='Underlying Distribution', histtype='step')
    # plt.errorbar(bin_center, x_vector_unf, yerr=sigma_x_unf, fmt='.', label='Unfolding')
    # plt.legend(loc='best')
    # plt.xlabel(r'True / Measured Value $x$')
    # plt.ylabel(r'# Events')

    return

    # raise NotImplementedError


energies = 1000.0 * np.random.power(0.70, 500000)
# energies = normal(loc=1000.0, scale=500, size=1000)
below_zero = energies < 0.0
energies[below_zero] = 1.0

detector = Detector(distribution='gaussian',
                    energy_loss='const',
                    make_noise=True,
                    smearing=True,
                    resolution_chamber=1.,
                    noise=0.,
                    plot=False)

signal, true_hits, detector_matrix = detector.simulate(energies)
matrix_inverse_unfolding(signal, true_hits, detector_matrix)


def svd_unfolding(signal, true_energy, detector_response_matrix, num_bins=20):
    u, s, v = np.linalg.svd(detector_response_matrix, full_matrices=True)
    print("U:\n" + str(u))
    print("S:\n" + str(s))
    print("V:\n" + str(v))
    raise NotImplementedError


def llh_unfolding(signal, true_energy, detector_response_matrix, num_bins=20):
    def LLH(f, data):
        return np.sum(np.log(f * powerlaw.pdf(data) + (1 - f) * powerlaw.pdf(data)))

    raise NotImplementedError
