import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def matrix_inverse_unfolding(signal, true_energy, detector_response_matrix, num_bins=20):
    # TODO: Should not base on [1] for either, most data ignored then
    y_vector = np.bincount(signal[1], minlength=num_bins - 1)
    x_vector = np.bincount(true_energy[1], minlength=num_bins - 1)

    # Get the inverse of the detector response matrix
    inv_detector_response_matrix = np.linalg.inv(detector_response_matrix)

    x_vector_unf = np.dot(inv_detector_response_matrix, y_vector)

    # Error propagation
    V_y = np.diag(y_vector)
    V_x_est = np.dot(inv_detector_response_matrix, np.dot(V_y, inv_detector_response_matrix.T))
    sigma_x_unf = np.sqrt(np.diag(V_x_est))

    print('x_unf   \t\t= %s' % str(np.round(x_vector_unf, 2)))
    print('simga_x_unf \t\t= %s' % str(np.round(sigma_x_unf, 2)))
    # Need to compare to underlying PDF, which can just be the counts
    # TODO: Change x_vector to the underlying distribution (either from Detector class, or find here)
    print('(unf - pdf) / sigma_x \t= %s ' % str(np.round((x_vector_unf - x_vector) / sigma_x_unf, 2)))

    plt.hist(bin_center, bins=num_bins, weights=x_vector, label='Underlying Distribution', histtype='step')
    plt.errorbar(bin_center, x_vector_unf, yerr=sigma_x_unf, fmt='.', label='Unfolding')
    plt.legend(loc='best')
    plt.xlabel(r'True / Measured Value $x$')
    plt.ylabel(r'# Events')

    raise NotImplementedError


def svd_unfolding(signal, true_energy, detector_response_matrix, num_bins=20):

    u, s, v = np.linalg.svd(detector_response_matrix, full_matrices=True)
    
    raise NotImplementedError
