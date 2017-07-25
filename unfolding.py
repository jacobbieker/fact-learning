import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import powerlaw
from detector import Detector
import numdifftools as nd
from pprint import pprint


def eigenvalue_cutoff(signal, true_energy, detector_matrix, unfolding_error):
    """
    Remove the lower eigenvalues that fall below the unfolding error, to smooth out the result
    :param signal: The signal from the detector
    :param true_energy: The true energy spectrum
    :param detector_matrix: The detector response matrix
    :param unfolding_error: The error in the unfolding, V_x
    :return:
    """
    inv_detector_matrix = np.linalg.inv(detector_matrix)
    eigen_vals, eigen_vecs = np.linalg.eig(detector_matrix)
    inv_eigen_vals, inv_eigen_vecs = np.linalg.eig(inv_detector_matrix)

    # Here, need to make the UDU^T set of matricies. the U is the eigenvectors of A, the response matrix
    # And D is the diagnol matrix with the members of the diagonal being the eigenvalues of A in decreasing
    # Order. So need to sort eigenvalues and put the array in a square matrix

    U = eigen_vecs
    eigen_vals = np.absolute(eigen_vals)
    sorting = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[sorting]
    D = np.diag(eigen_vals)
    kappa = max(eigen_vals)/min(eigen_vals)
    print("Kappa:\n", str(kappa))

    sum_signal_per_chamber = np.sum(signal, axis=1) # The x value
    y_vector = np.histogram(sum_signal_per_chamber, bins=detector_matrix.shape[0])
    x_vector_true = np.histogram(true_energy, bins=detector_matrix.shape[0])
    print("Stop")
    c = np.dot(x_vector_true[0], U.T)
    b = np.dot(y_vector[0], U.T)
    d_b = np.dot(D, b)

    raise NotImplementedError


def matrix_inverse_unfolding(signal, true_energy, detector_response_matrix, num_bins=50):
    sum_signal_per_chamber = np.sum(signal, axis=1)

    # x_vector =
    detector_matrix_col = detector_response_matrix
    y_vector = np.histogram(sum_signal_per_chamber,
                            bins=np.linspace(min(sum_signal_per_chamber), max(sum_signal_per_chamber),
                                             detector_matrix_col.shape[0]))
    y_vector = np.histogram(sum_signal_per_chamber, bins=detector_matrix_col.shape[0])

    # Get the inverse of the detector response matrix
    inv_detector_response_matrix = np.linalg.inv(detector_matrix_col)

    x_vector_unf = np.dot(inv_detector_response_matrix, y_vector[0])

    print("Unfolded Size:\n", str(len(x_vector_unf)))

    # Error propagation
    V_y = np.diag(y_vector[0])
    V_x_est = np.dot(inv_detector_response_matrix, np.dot(V_y, inv_detector_response_matrix.T))
    sigma_x_unf = np.sqrt(np.diag(V_x_est))

    print('x_unf   \t\t= %s' % str(np.round(x_vector_unf, 2)))
    print('simga_x_unf \t\t= %s' % str(np.round(sigma_x_unf, 2)))
    # Need to compare to underlying PDF, which can just be the counts
    # TODO: Change x_vector to the underlying distribution (either from Detector class, or find here)
    # print('(unf - pdf) / sigma_x \t= %s ' % str(np.round((x_vector_unf - x_vector) / sigma_x_unf, 2)))

    # So current problems are that as teh number of events goes up, the unfolded amount of energy increases for the final
    # result. Like, instead of increasing the amount of counts for energies at 1000, it increaseas the total energy, but
    # Only has a few particles in each spot. Maybe another issue with binning? Or my unfolding is really that bad, when
    # taken from what Mathis gave me. Doesn't make sense, some stupid thing I'm doing is messing this up.


    return

    # raise NotImplementedError


def svd_unfolding(signal, true_energy, detector_response_matrix, num_bins=20):
    u, s, v = np.linalg.svd(detector_response_matrix, full_matrices=True)
    print("U:\n" + str(u))
    print("S:\n" + str(s))
    print("V:\n" + str(v))
    # plt.imshow(s, interpolation="nearest", origin="upper")
    # plt.colorbar()
    # plt.title("S Matrix")
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()
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


energies = 1000.0 * np.random.power(0.70, 5000)
# energies = normal(loc=1000.0, scale=500, size=1000)
below_zero = energies < 0.0
energies[below_zero] = 1.0

detector = Detector(distribution='gaussian',
                    energy_loss='random',
                    make_noise=False,
                    smearing=False,
                    resolution_chamber=1.,
                    noise=0.,
                    plot=False)

detector_test = Detector(distribution='gaussian',
                         energy_loss='random',
                         make_noise=False,
                         smearing=False,
                         resolution_chamber=1.,
                         noise=0.,
                         plot=False)

test_signal, test_true_hits, test_energies, test_detector_matrix = detector_test.simulate(energies)
signal, true_hits, energies, detector_matrix = detector.simulate(energies)
eigenvalue_cutoff(signal, energies, detector_matrix, 0.0)
#test_unfolding(energies, detector_matrix, num_bins=15)
# svd_unfolding(test_signal, energies, detector_matrix)
matrix_inverse_unfolding(test_signal, energies, detector_matrix, num_bins=15)
