import numpy as np
from scipy.stats import powerlaw
import numdifftools as nd

import evaluate_unfolding


def obtain_coefficients(signal, true_energy, eigen_values, eigen_vectors, cutoff=False):
    U = eigen_vectors
    eigen_vals = np.absolute(eigen_values)
    sorting = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[sorting]
    D = np.diag(eigen_vals)

    sum_signal_per_chamber = np.sum(signal, axis=1) # The x value
    y_vector = np.histogram(sum_signal_per_chamber, bins=U.shape[0])
    x_vector_true = np.histogram(true_energy, bins=U.shape[0])
    c = np.dot(U.T, y_vector[0])
    b = np.dot(U.T, x_vector_true[0])
    d_b = np.dot(D, b)

    # Now to do the unfolding by dividing coefficients by the eigenvalues in D to get b_j
    b_j = np.zeros_like(c)
    for j, coefficient in enumerate(c):
        # Cutting the number of values in half, just to test it
        if cutoff:
            if j < (D.shape[0]/2):
                b_j[j] = coefficient / D[j,j]
        else:
            b_j[j] = coefficient / D[j,j]

    unfolded_x = np.dot(U, b_j)

    return b, b_j, c


def eigenvalue_cutoff(signal, true_energy, detector_matrix, unfolding_error, cutoff=False):
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
    c = np.dot(U.T, y_vector[0])
    b = np.dot(U.T, x_vector_true[0])
    d_b = np.dot(D, b)

    # Now to do the unfolding by dividing coefficients by the eigenvalues in D to get b_j
    b_j = np.zeros_like(c)
    for j, coefficient in enumerate(c):
        # Cutting the number of values in half, just to test it
        if cutoff:
            if j < (D.shape[0]/2):
                b_j[j] = coefficient / D[j,j]
        else:
            b_j[j] = coefficient / D[j,j]
    unfolded_x = np.dot(U, b_j)
    print(unfolded_x)

    return eigen_vals, eigen_vecs


def matrix_inverse_unfolding(signal, detector_response_matrix):
    """
    Unfold the signal using simple matrix unfolding
    :param signal: The signal from the detector, in either total energy per chamber or energy per particle per chamber
    :param detector_response_matrix: The detector Response Matrix, normalized by column
    :return: The unfolded signal, sigma in the unfolding, the x error estimation, y error, and the unf - pdf / sigma_x
    """
    if signal.ndim == 2:
        sum_signal_per_chamber = np.sum(signal, axis=1)
        y_vector = np.histogram(sum_signal_per_chamber, bins=detector_response_matrix.shape[0])
    else:
        y_vector = [signal, 0]

    x_pdf_space = np.linspace(powerlaw.ppf(0.01, 0.70), powerlaw.ppf(1.0, 0.70), detector_response_matrix.shape[0])
    x_vector = powerlaw.pdf(x_pdf_space, 0.70)

    # Get the inverse of the detector response matrix
    inv_detector_response_matrix = np.linalg.inv(detector_response_matrix)

    x_vector_unf = np.dot(y_vector[0], inv_detector_response_matrix)

    # Error propagation
    V_y = np.diag(y_vector[0])
    V_x_est = np.dot(inv_detector_response_matrix, np.dot(V_y, inv_detector_response_matrix.T))
    sigma_x_unf = np.sqrt(np.diag(V_x_est))

    #print('x_unf   \t\t= %s' % str(np.round(x_vector_unf, 2)))
    #print('simga_x_unf \t\t= %s' % str(np.round(sigma_x_unf, 2)))
    #print('(unf - pdf) / sigma_x \t= %s ' % str(np.round((x_vector_unf - x_vector) / sigma_x_unf, 2)))

    unf_pdf_sigma = (x_vector_unf - x_vector) / sigma_x_unf
    return x_vector_unf, sigma_x_unf, V_x_est, V_y, unf_pdf_sigma


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

    # So the USV*x = USV*true_energy = signal
    # So need to undo that
    # Need to bin tru energy and signal
    z = np.dot(v.T, true_energy)
    d = np.dot(u.T, signal)

    # d_i = s_iz_i so z_i = d_i/s_i
    z_i = np.zeros_like(s.shape[0])
    for index, i, in enumerate(d):
        z_i[index] = d[i,i] / s[i,i]

    #Now do it with V to get the unfolded distrubtion
    unfolded_signal = np.dot(v, z_i)
    print("Differences (Should maybe be 0):")
    print(unfolded_signal - true_energy)

    # And so x = Vz, but only if you know true_energy beforehand
    true_unfolded_x = np.dot(v, z)
    print("Differences (Should be Zero):")
    print(true_unfolded_x - true_energy)

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
