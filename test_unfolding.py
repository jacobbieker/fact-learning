import numpy as np
from matplotlib import pyplot as plt
from detector import Detector
from unfolding import matrix_inverse_unfolding, obtain_coefficients, svd_unfolding, llh_unfolding, eigenvalue_cutoff
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
    llh_unfolding(sum_signal_per_chamber, y_vector[0], detector_matrix_col, tau=1)

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
        A[0, 0] = 1. - epsilon
        A[0, 1] = epsilon
        A[-1, -1] = 1. - epsilon
        A[-1, -2] = epsilon
        for i in range(num_bins)[1:-1]:
            A[i, i] = 1. - 2. * epsilon
            A[i, i + 1] = epsilon
            A[i, i - 1] = epsilon
        return A

    def get_unbalance_response_matrix(norm='column'):
        if num_bins < 2:
            raise ValueError('dim must be larger than 2')
        A = np.zeros((num_bins, num_bins))
        A[0, 0] = 1.
        A[-1, -1] = 1.
        for i in range(num_bins)[1:-1]:
            subtract = np.random.random()
            A[i, i] = 1. - subtract
            if norm == 'column':
                A[i, i + 1] = 1. - (1. - subtract / 2.)
                A[i, i - 1] = 1. - (1. - subtract / 2.)
            elif norm == 'row':
                A[i + 1, i] = 1. - (1. - subtract / 2.)
                A[i - 1, i] = 1. - (1. - subtract / 2.)
        if norm == 'column':
            A = A / A.sum(axis=0, keepdims=True)
        else:
            A = A / A.sum(axis=1, keepdims=True)
        return A

    detector_response_matrix = get_response_matrix()
    col_norm_matrix = get_unbalance_response_matrix()
    row_norm_matrix = get_unbalance_response_matrix('row')
    y_vector = np.histogram(energies, bins=num_bins)
    detected_signal = np.dot(y_vector[0], detector_response_matrix)
    col_detected_signal = np.dot(y_vector[0], col_norm_matrix)
    row_detected_signal = np.dot(y_vector[0], row_norm_matrix)
    col_unfolding_results = matrix_inverse_unfolding(col_detected_signal, col_norm_matrix)
    row_unfolding_results = matrix_inverse_unfolding(row_detected_signal, row_norm_matrix)
    matrix_unfolding_results = matrix_inverse_unfolding(detected_signal, detector_response_matrix)
    if epsilon == 0.0:
        assert y_vector[0].all() == matrix_unfolding_results[0].all()
        assert y_vector[0].all() == col_unfolding_results[0].all()
        assert y_vector[0].all() == row_unfolding_results[0].all()
    if plot:
        evaluate_unfolding.plot_unfolded_vs_true(y_vector, matrix_unfolding_results[0], energies, num_bins=num_bins)
        print("True x: " + str(y_vector[0]))
        print("Difference: " + str(y_vector[0] - matrix_unfolding_results[0]))
        print("Difference: " + str(y_vector[0] - col_unfolding_results[0]))
        print("Difference: " + str(y_vector[0] - row_unfolding_results[0]))


def test_detector_response_matrix_unfolding(random_state=None, noise=True, smearing=True, plot=False):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    energies = 1000.0 * random_state.power(0.70, 50000)
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

    matrix_inverse_unfolding_results = matrix_inverse_unfolding(signal, detector_matrix)

    if plot:
        evaluate_detector.plot_response_matrix(detector_matrix)
        sum_signal_per_chamber = np.sum(signal, axis=1)
        y_vector = np.histogram(sum_signal_per_chamber, bins=detector_matrix.shape[0])
        print("True x: " + str(y_vector[0]))
        print("Difference: " + str(y_vector[0] - matrix_inverse_unfolding_results[0]))
        evaluate_unfolding.plot_unfolded_vs_true(matrix_inverse_unfolding_results[0], energies_return,
                                                 errors=matrix_inverse_unfolding_results[1], title="Matrix Unfolding")


def test_multiple_datasets_std(random_state=None, method=matrix_inverse_unfolding, num_datasets=20, num_bins=20,
                               noise=True, smearing=True, plot=False):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    # Array to hold the arrays of means and stuff
    array_of_unfolding_errors = []

    subset_random_state_seed = random_state.random_integers(low=3000000, size=num_datasets)

    # Generate the different datasets based off the random_state
    for seed in subset_random_state_seed:
        subset_random_state = np.random.RandomState(seed)

        # Now generate the datasets
        energies = 1000.0 * subset_random_state.power(0.70, 500)
        below_zero = energies < 1.0
        energies[below_zero] = 1.0

        detector = Detector(distribution='gaussian',
                            energy_loss='const',
                            make_noise=noise,
                            smearing=smearing,
                            resolution_chamber=1.,
                            noise=0.,
                            response_bins=num_bins,
                            random_state=subset_random_state)

        signal, true_hits, energies_return, detector_matrix = detector.simulate(energies)

        if method == matrix_inverse_unfolding:
            matrix_inverse_unfolding_results = method(signal, detector_matrix)
            array_of_unfolding_errors.append(matrix_inverse_unfolding_results[4])
        else:
            matrix_inverse_unfolding_results = method(signal, detector_matrix)
            array_of_unfolding_errors.append(matrix_inverse_unfolding_results[0])

    array_of_unfolding_errors = np.asarray(array_of_unfolding_errors)

    # Different ways of getting the mean and std
    # Mean and std of datasets
    # print(np.mean(array_of_unfolding_errors, axis=1))
    # print(np.std(array_of_unfolding_errors, axis=1))

    # Mean and std of the whole thing
    print("Mean (All): " + str(np.mean(array_of_unfolding_errors)))
    print("Std (All): " + str(np.std(array_of_unfolding_errors)))

    if plot:
        evaluate_unfolding.plot_error_stats(np.mean(array_of_unfolding_errors, axis=1),
                                            np.std(array_of_unfolding_errors, axis=1))
        # evaluate_unfolding.plot_error_stats(np.mean(array_of_unfolding_errors), np.std(array_of_unfolding_errors))


def test_eigenvalue_cutoff_response_matrix_unfolding(random_state=None, cutoff=5, num_bins=20, plot=False):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    energies = 1000.0 * random_state.power(0.70, 50000)
    below_zero = energies < 1.0
    energies[below_zero] = 1.0

    detector = Detector(distribution='gaussian',
                        energy_loss='const',
                        make_noise=True,
                        smearing=True,
                        resolution_chamber=1.,
                        noise=0.,
                        response_bins=num_bins,
                        random_state=random_state)
    signal, true_hits, energies_return, detector_matrix = detector.simulate(
        energies)
    eigenvalue_cutoff_results = eigenvalue_cutoff(signal, energies, detector_matrix, 0.0, cutoff=cutoff)
    eigenvalues, eigenvectors = eigenvalue_cutoff_results[0], eigenvalue_cutoff_results[1]

    true, folded, measured = obtain_coefficients(signal, energies, eigenvalues, eigenvectors, cutoff=cutoff)
    if true_hits.ndim == 2:
        sum_true_energy = np.sum(true_hits, axis=1)
        true_hits = np.histogram(sum_true_energy, bins=detector_matrix.shape[0])

    if plot:
        evaluate_unfolding.plot_eigenvalue_coefficients(true, folded, measured, eigenvalue_cutoff_results[6])
        # evaluate_unfolding.plot_eigenvalues(eigenvalues, eigenvectors, n_dims=detector_matrix.shape[0])
        evaluate_unfolding.plot_unfolded_vs_true(eigenvalue_cutoff_results[2], energies_return,
                                                 errors=eigenvalue_cutoff_results[6],
                                                 title="Unfolding X")
        evaluate_unfolding.plot_unfolded_vs_true(eigenvalue_cutoff_results[3], energies_return,
                                                 errors=eigenvalue_cutoff_results[6],
                                                 title="Unfolding X Other")
        evaluate_unfolding.plot_unfolded_vs_true(eigenvalue_cutoff_results[4], energies_return,
                                                 errors=eigenvalue_cutoff_results[6],
                                                 title="Unfolding True")
        evaluate_unfolding.plot_unfolded_vs_true(eigenvalue_cutoff_results[5], energies_return,
                                                 errors=eigenvalue_cutoff_results[6],
                                                 title="Unfolding True 2")


def test_svd_unfolding(random_state=None, smearing=True, noise=True, num_bins=20, plot=False, cutoff=None):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    energies = 1000.0 * random_state.power(0.70, 5000)
    below_zero = energies < 1.0
    energies[below_zero] = 1.0

    detector = Detector(distribution='gaussian',
                        energy_loss='const',
                        make_noise=noise,
                        smearing=smearing,
                        resolution_chamber=1.,
                        noise=0.,
                        response_bins=num_bins,
                        rectangular_bins=20,
                        random_state=random_state)

    signal, true_hits, energies_return, detector_matrix = detector.simulate(
        energies)

    svd_unfolding_results = svd_unfolding(signal, detector_matrix, cutoff=cutoff)

    if true_hits.ndim == 2:
        sum_true_energy = np.sum(true_hits, axis=1)
        true_hits = np.histogram(sum_true_energy, bins=detector_matrix.shape[0])

    print(svd_unfolding_results[0])
    print("Differences:")
    print(svd_unfolding_results[0] - true_hits[0])
    print("SVD Sum:")
    print(np.sum(svd_unfolding_results[0]))

    if plot:
        evaluate_unfolding.plot_unfolded_vs_true(svd_unfolding_results[0], energies_return,
                                                 title="SVD Unfolding")
        evaluate_unfolding.plot_svd_parts(svd_unfolding_results[1], svd_unfolding_results[2], svd_unfolding_results[3])


def test_epsilon_svd_unfolding(random_state=None, epsilon=0.2, num_row=10, num_col=20, plot=False):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    energies = 1000.0 * random_state.power(0.70, 500)
    below_zero = energies < 1.0
    energies[below_zero] = 1.0

    # TODO: Fill rectangular matrix correct, currently making square matrix with extra zeros

    def get_response_matrix():
        A = np.zeros((num_row, num_col))
        A[0, 0] = 1. - epsilon
        A[0, 1] = epsilon
        A[-1, -1] = 1. - epsilon
        A[-1, -2] = epsilon
        for i in range(min(num_row, num_col))[1:-1]:
            A[i, i] = 1. - 2. * epsilon
            A[i, i + 1] = epsilon
            A[i, i - 1] = epsilon
        return A

    def get_unbalance_response_matrix(norm='column'):
        A = np.zeros((num_row, num_col))
        A[0, 0] = 1.
        A[-1, -1] = 1.
        for i in range(min(num_row, num_col))[1:-1]:
            subtract = np.random.random()
            A[i, i] = 1. - subtract
            if norm == 'column':
                A[i, i + 1] = 1. - (1. - subtract / 2.)
                A[i, i - 1] = 1. - (1. - subtract / 2.)
            elif norm == 'row':
                A[i + 1, i] = 1. - (1. - subtract / 2.)
                A[i - 1, i] = 1. - (1. - subtract / 2.)
        if norm == 'column':
            A = A / A.sum(axis=0, keepdims=True)
        else:
            A = A / A.sum(axis=1, keepdims=True)
        return A

    detector_response_matrix = get_response_matrix()
    col_norm_matrix = get_unbalance_response_matrix()
    row_norm_matrix = get_unbalance_response_matrix('row')
    y_vector = np.histogram(energies, bins=min(num_row, num_col))
    detected_signal = np.dot(y_vector[0], detector_response_matrix)
    col_detected_signal = np.dot(y_vector[0], col_norm_matrix)
    row_detected_signal = np.dot(y_vector[0], row_norm_matrix)
    col_unfolding_results = svd_unfolding(col_detected_signal, col_norm_matrix)
    row_unfolding_results = svd_unfolding(row_detected_signal, row_norm_matrix)
    matrix_unfolding_results = svd_unfolding(detected_signal, detector_response_matrix)
    if epsilon == 0.0:
        assert y_vector[0].all() == matrix_unfolding_results[0].all()
        assert y_vector[0].all() == col_unfolding_results[0].all()
        assert y_vector[0].all() == row_unfolding_results[0].all()
    if plot:
        evaluate_unfolding.plot_unfolded_vs_true(y_vector, matrix_unfolding_results[0], energies,
                                                 num_bins=min(num_row, num_col))
        print("True x: " + str(y_vector[0]))
        print("Difference: " + str(y_vector[0] - matrix_unfolding_results[0]))
        print("Difference: " + str(y_vector[0] - col_unfolding_results[0]))
        print("Difference: " + str(y_vector[0] - row_unfolding_results[0]))


def test_llh_unfolding(random_state=None, tau=1, unfolding=True, num_bins=20, noise=True, smearing=True,
                       regularized=True, noise_val=0., resolution_val=1., plot=False):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    energies = 1000.0 * random_state.power(0.70, 5000)
    below_zero = energies < 1.0
    energies[below_zero] = 1.0

    detector = Detector(distribution='gaussian',
                        energy_loss='const',
                        make_noise=noise,
                        smearing=smearing,
                        resolution_chamber=resolution_val,
                        noise=noise_val,
                        response_bins=num_bins,
                        rectangular_bins=num_bins,
                        random_state=random_state)

    signal, true_hits, energies_return, detector_matrix = detector.simulate(
        energies)

    llh_unfolding_results = llh_unfolding(signal, energies_return, detector_matrix, tau=tau, unfolding=unfolding,
                                          regularized=regularized, num_bins=num_bins)

    if true_hits.ndim == 2:
        sum_true_energy = np.sum(true_hits, axis=1)
        true_hits = np.histogram(sum_true_energy, bins=detector_matrix.shape[0])

    if plot:
        evaluate_unfolding.plot_unfolded_vs_signal_vs_true(llh_unfolding_results[0], llh_unfolding_results[1],
                                                           llh_unfolding_results[2])

        # evaluate_unfolding.plot_unfolded_vs_true(llh_unfolding_results, energies_return,
        #                                        title="LLH Unfolding")


if __name__ == "__main__":
    test_llh_unfolding(1347, tau=0.005, plot=True, regularized=False, smearing=True, noise=True, noise_val=0000000.,
                       resolution_val=1., unfolding=True)
    test_llh_unfolding(np.random.RandomState(), tau=0.005, plot=True, regularized=False, smearing=False, noise=False, noise_val=0.,
                       resolution_val=1., unfolding=True)
    # test_identity_response_matrix_unfolding(1347, )
    # test_svd_unfolding(1347, plot=False)
    # test_epsilon_svd_unfolding(1347, plot=True)
    # test_multiple_datasets_std(1347, method=matrix_inverse_unfolding, smearing=False, noise=False, plot=True, num_datasets=500)
    # test_multiple_datasets_std(1347, method=svd_unfolding, smearing=False, noise=False, plot=True, num_datasets=500)
    # test_detector_response_matrix_unfolding(1347, plot=True)
    # test_eigenvalue_cutoff_response_matrix_unfolding(1347, cutoff=15, num_bins=20, plot=True)
    # test_eigenvalue_cutoff_response_matrix_unfolding(1347, cutoff=10, num_bins=20, plot=True)
    # test_identity_response_matrix_unfolding(1347, plot=False)
    # test_epsilon_response_matrix_unfolding(1347, epsilon=0.0, num_bins=20, plot=True)
    # test_epsilon_response_matrix_unfolding(1347, epsilon=0.2, num_bins=600, plot=True)
    # test_epsilon_response_matrix_unfolding(1347, epsilon=0.499, num_bins=600, plot=True)
