import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import powerlaw
from detector import Detector
import numdifftools as nd
from pprint import pprint


def matrix_inverse_unfolding(signal, true_energy, detector_response_matrix, num_bins=50):
    sum_signal_per_chamber = np.sum(signal, axis=1)

    # x_vector =
    detector_matrix, detector_matrix_col, detector_matrix_row = detector_response_matrix
    y_vector = np.histogram(sum_signal_per_chamber, bins=np.linspace(min(sum_signal_per_chamber), max(sum_signal_per_chamber), detector_matrix_col.shape[0]))
    y_vector = np.histogram(sum_signal_per_chamber, bins=detector_matrix.shape[0])
    print(y_vector)
    plt.bar(y_vector[1][:-1], y_vector[0], width=y_vector[1][1:], label="Y Vector")
    plt.hist(true_energy, bins=np.linspace(min(true_energy), max(true_energy), num_bins), normed=False,
             label="True Energy", histtype='step')
    plt.hist(sum_signal_per_chamber, bins=np.linspace(min(sum_signal_per_chamber), max(sum_signal_per_chamber), num_bins), normed=False,
             label="Summed Signal Energy", histtype='step')
    # Problem is that it is already binned, so this is just binning it again. Still not sure why it gets so large
    # But for it being small, probably because its a bin of bins, so that's why its at 1 or 4 or things like that
    plt.title("True Energy vs Y_Vector")
    plt.legend(loc='best')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    print(len(y_vector[0]))

    # Get the inverse of the detector response matrix
    inv_detector_response_matrix = np.linalg.inv(detector_matrix_col)

    #Check if its the identity
    print(np.allclose(np.dot(inv_detector_response_matrix, detector_matrix_col), np.eye(detector_matrix_col.shape[0])))
    # Gives the correct response now, wasn't before

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
    #print('(unf - pdf) / sigma_x \t= %s ' % str(np.round((x_vector_unf - x_vector) / sigma_x_unf, 2)))

    plt.hist(x_vector_unf, bins=num_bins, label="Unfolded Energy")
    plt.hist(true_energy, bins=np.linspace(min(true_energy), max(true_energy), num_bins), normed=False,
             label="True Energy", histtype='step')
    y_values = np.histogram(x_vector_unf, bins=len(x_vector_unf))
    #plt.errorbar(x_vector_unf, y=y_values[0], yerr=sigma_x_unf)
    plt.title("Number of Particles: " + str(true_energy.shape[0]))
    plt.legend(loc='best')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    # plt.hist(bin_center, bins=num_bins, weights=x_vector, label='Underlying Distribution', histtype='step')
    # plt.errorbar(bin_center, x_vector_unf, yerr=sigma_x_unf, fmt='.', label='Unfolding')
    # plt.legend(loc='best')
    # plt.xlabel(r'True / Measured Value $x$')
    # plt.ylabel(r'# Events')

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
    #plt.imshow(s, interpolation="nearest", origin="upper")
    #plt.colorbar()
    #plt.title("S Matrix")
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.show()
    raise NotImplementedError


def test_unfolding(true_energy, detector_response_matrix, num_bins=20):
    detector_matrix, detector_matrix_col, detector_matrix_row = detector_response_matrix
    detector_matrix_col = np.eye(num_bins)
    print(detector_matrix_col)
    y_vector = np.histogram(true_energy, bins=num_bins)
    inv_detector_matrix = np.linalg.inv(detector_matrix_col)
    print(inv_detector_matrix)
    sum_signal_per_chamber = np.dot(inv_detector_matrix, y_vector[0])
    print(sum_signal_per_chamber == y_vector[0])
    eigen_vals, eigen_vecs = np.linalg.eig(detector_matrix_col)
    inv_eigen_vals, inv_eigen_vecs = np.linalg.eig(inv_detector_matrix)
    print(len(y_vector[0]))
    print(len(sum_signal_per_chamber))
    sum_signal_per_chamber = np.ndarray.astype(sum_signal_per_chamber, np.int64)
    plt.bar(y_vector[1][:-1], y_vector[0], width=y_vector[1][1:], label="Y Vector")
    plt.hist(true_energy, bins=np.linspace(min(true_energy), max(true_energy), num_bins), normed=False,
             label="True Energy", histtype='step')
    plt.bar(y_vector[1][:-1], sum_signal_per_chamber, width=y_vector[1][1:], label="Unfolded Energy") #bins=np.linspace(min(sum_signal_per_chamber), max(sum_signal_per_chamber), num_bins), normed=False,
             #label="Summed Signal Energy", histtype='step')
    plt.title("True Energy vs Y_Vector")
    plt.legend(loc='best')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()
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
test_unfolding(energies, detector_matrix, num_bins=15)
#svd_unfolding(test_signal, energies, detector_matrix)
matrix_inverse_unfolding(test_signal, energies, detector_matrix)

