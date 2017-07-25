import numpy as np
from matplotlib import pyplot as plt
from detector import Detector
from unfolding import matrix_inverse_unfolding, svd_unfolding, llh_unfolding, eigenvalue_cutoff
import evaluate_unfolding
import evaluate_detector


def test_identity_response_matrix_unfolding(random_state=None, num_bins=20, plot=False):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    energies = 1000.0 * random_state.power(0.70, 500)
    below_zero = energies < 0.0
    energies[below_zero] = 1.0

    detector_matrix_col = np.eye(num_bins)
    y_vector = np.histogram(energies, bins=num_bins)
    inv_detector_matrix = np.linalg.inv(detector_matrix_col)
    sum_signal_per_chamber = np.dot(inv_detector_matrix, y_vector[0])
    sum_signal_per_chamber = np.ndarray.astype(sum_signal_per_chamber, np.int64)

    if plot:
        plt.bar(y_vector[1][:-1], y_vector[0], width=y_vector[1][1:], label="Y Vector")
        plt.hist(energies, bins=np.linspace(min(energies), max(energies), num_bins), normed=False,
                 label="True Energy", histtype='step')
        plt.bar(y_vector[1][:-1], sum_signal_per_chamber, width=y_vector[1][1:],
                label="Unfolded Energy")
        plt.title("True Energy vs Y_Vector")
        plt.legend(loc='best')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

    assert sum_signal_per_chamber.all() == y_vector[0].all()
    assert np.allclose(np.dot(inv_detector_matrix, detector_matrix_col), np.eye(detector_matrix_col.shape[0]))


def test_epsilon_response_matrix_unfolding(random_state=None, epsilon=0.2, num_bins=20, plot=False):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    energies = 1000.0 * random_state.power(0.70, 500)
    below_zero = energies < 1.0
    energies[below_zero] = 1.0

    def get_response_matrix():
        if num_bins < 2:
            raise ValueError("'dim' must be larger than 2")
        A = np.zeros((num_bins, num_bins))
        A[0, 0] = 1.-epsilon
        A[0, 1] = epsilon
        A[-1, -1] = 1.-epsilon
        A[-1, -2] = epsilon
        for i in range(num_bins)[1:-1]:
            A[i, i] = 1.-2.*epsilon
            A[i, i+1] = epsilon
            A[i, i-1] = epsilon
        return A

    detector_response_matrix = get_response_matrix()
    y_vector = np.histogram(energies, bins=num_bins)
    detected_signal = np.dot(y_vector[0], detector_response_matrix)
    matrix_unfolding_results = matrix_inverse_unfolding(detected_signal, energies, detector_response_matrix, num_bins=num_bins)
    if epsilon == 0.0:
        assert y_vector[0].all() == matrix_unfolding_results[0].all()
    else:
        print(y_vector[0])
        print(matrix_unfolding_results[0])
    if plot:
        evaluate_unfolding.plot_unfolded_vs_true(y_vector, matrix_unfolding_results[0], energies, num_bins=num_bins)


def test_detector_response_matrix_unfolding(random_state=None, noise=True, smearing=True, plot=False):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    energies = 1000.0 * random_state.power(0.70, 500)
    # energies = normal(loc=1000.0, scale=500, size=1000)
    below_zero = energies < 1.0
    energies[below_zero] = 1.0

    detector = Detector(distribution='gaussian',
                        energy_loss='const',
                        make_noise=noise,
                        smearing=smearing,
                        resolution_chamber=1.,
                        noise=0.,
                        random_state=random_state)

    signal, true_hits, energies_return, detector_matrix = detector.simulate(
        energies)

    matrix_inverse_unfolding_results = matrix_inverse_unfolding(signal, true_hits, detector_matrix)
    if plot:
        evaluate_detector.plot_response_matrix(detector_matrix)
        sum_signal_per_chamber = np.sum(signal, axis=1)
        y_vector = np.histogram(sum_signal_per_chamber, bins=detector_matrix.shape[0])
        evaluate_unfolding.plot_unfolded_vs_true(y_vector, matrix_inverse_unfolding_results[0], energies_return)


def test_eigenvalue_cutoff_response_matrix_unfolding(random_state=None, epsilon=0.2, num_bins=20, plot=False):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    energies = 1000.0 * random_state.power(0.70, 500)
    below_zero = energies < 1.0
    energies[below_zero] = 1.0

    detector = Detector(distribution='gaussian',
                        energy_loss='const',
                        make_noise=True,
                        smearing=True,
                        resolution_chamber=1.,
                        noise=0.,
                        random_state=random_state)
    signal, true_hits, energies_return, detector_matrix = detector.simulate(
        energies)
    eigenvalues, eigenvectors = eigenvalue_cutoff(signal, energies, detector_matrix, 0.0)
    evaluate_unfolding.plot_eigenvalues(eigenvalues, eigenvectors, n_dims=detector_matrix.shape[0])


if __name__ == "__main__":
    test_detector_response_matrix_unfolding(1347, plot=True)
    test_eigenvalue_cutoff_response_matrix_unfolding(1347, plot=False)
    #test_identity_response_matrix_unfolding(1347, plot=True)
    #test_epsilon_response_matrix_unfolding(1347, epsilon=0.0, plot=True)
    #test_epsilon_response_matrix_unfolding(1347, epsilon=0.2, plot=True)
    #test_epsilon_response_matrix_unfolding(1347, epsilon=0.495, plot=True)
