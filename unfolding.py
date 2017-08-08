import numpy as np
from scipy.stats import powerlaw
import numdifftools as nd
import math
import matplotlib.pyplot as plt
import numpy
from pprint import pprint


def obtain_coefficients(signal, true_energy, eigen_values, eigen_vectors, cutoff=None):
    U = eigen_vectors
    eigen_vals = np.absolute(eigen_values)
    sorting = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[sorting]
    D = np.diag(eigen_vals)

    sum_signal_per_chamber = np.sum(signal, axis=1)  # The x value
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
            if j < cutoff:
                # print(D[j, j])
                b_j[j] = coefficient / D[j, j]
            else:
                b_j[j] = 0.0
        else:
            b_j[j] = coefficient / D[j, j]

    unfolded_x = np.dot(U, b_j)

    return b, b_j, c


def eigenvalue_cutoff(signal, true_energy, detector_matrix, unfolding_error, cutoff=None):
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

    # Sort on U as well
    # Eigenvalues are set by iterating through bi, ci, and multiplying by 1/lambda
    # Cutoff is setting it to 0
    # Set the eigenvalues to zero after the inverse, so basically infinity on the non-inverse eigenvalues
    # So use for loop for it, setting everything after the cutoff to 0
    # If done right, eigenvalue cutoff will have less events, but total should have same amount
    # So the eigenvalues are the coefficeints of the c and/or b values, the transformed vectors
    # Can fix the decrease in number of events by multiply by the (true_number / detected number)
    # Problem with that fix though is that it of course gives back the same number, mathematically it has to
    eigen_vals = np.absolute(eigen_vals)
    sorting = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[sorting]
    U = eigen_vecs[sorting]
    # U = eigen_vecs
    D = np.diag(eigen_vals)
    kappa = max(eigen_vals) / min(eigen_vals)
    print("Kappa:\n", str(kappa))

    assert (np.isclose((U * eigen_vals).all(), detector_matrix.all()))

    sum_signal_per_chamber = np.sum(signal, axis=1)  # The x value
    y_vector = np.histogram(sum_signal_per_chamber, bins=detector_matrix.shape[0])
    x_vector_true = np.histogram(true_energy, bins=detector_matrix.shape[0])

    inv_U = np.linalg.inv(U)

    c = np.dot(inv_U, y_vector[0])
    b = np.dot(inv_U, x_vector_true[0])
    d_b = np.dot(D, b)

    # Now to do the unfolding by dividing coefficients by the eigenvalues in D to get b_j
    b_j = np.zeros_like(c)
    for j, coefficient in enumerate(c):
        # Cutting the number of values in half, just to test it
        if cutoff:
            if j < cutoff:
                print(D[j, j])
                b_j[j] = coefficient / D[j, j]
            else:
                b_j[j] = 0.00
        else:
            b_j[j] = coefficient / D[j, j]
    unfolded_x = np.dot(U, b_j)
    unfolded_x_other = np.dot(b_j, U)
    unfolded_multiplied = unfolded_x * (x_vector_true[0] / unfolded_x)
    unfolded_multiplied2 = unfolded_x_other * (x_vector_true[0] / unfolded_x_other)
    print(unfolded_x)
    print("Sums (unfolded_x, unfolded_x_other, multiplied, multiplied2):")
    print(np.sum(unfolded_x))
    print(np.sum(unfolded_x_other))
    print(np.sum(unfolded_multiplied))
    print(np.sum(unfolded_multiplied2))
    print("Difference (U * b_j):")
    print(unfolded_x - x_vector_true[0])
    print("Difference (b_j * U):")
    print(unfolded_x_other - x_vector_true[0])
    print("Difference (Multiplied):")
    print(unfolded_multiplied - x_vector_true[0])
    print("Difference (Multiplied2):")
    print(unfolded_multiplied2 - x_vector_true[0])

    # Error propagation

    # Get inverse of truncated "response matrix" U * eigenvalues
    inv_detector_response_matrix = np.linalg.inv(U * eigen_vals)
    V_y = np.diag(y_vector[0])
    V_x_est = np.dot(inv_detector_response_matrix, np.dot(V_y, inv_detector_response_matrix.T))
    sigma_x_unf = np.sqrt(np.diag(V_x_est))

    # print('x_unf   \t\t= %s' % str(np.round(x_vector_unf, 2)))
    # print('simga_x_unf \t\t= %s' % str(np.round(sigma_x_unf, 2)))
    # print('(unf - pdf) / sigma_x \t= %s ' % str(np.round((x_vector_unf - x_vector) / sigma_x_unf, 2)))

    unf_pdf_sigma = (unfolded_x - x_vector_true[0]) / sigma_x_unf

    return eigen_vals, U, unfolded_x, unfolded_x_other, unfolded_multiplied, unfolded_multiplied2, sigma_x_unf


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

    # print('x_unf   \t\t= %s' % str(np.round(x_vector_unf, 2)))
    # print('simga_x_unf \t\t= %s' % str(np.round(sigma_x_unf, 2)))
    # print('(unf - pdf) / sigma_x \t= %s ' % str(np.round((x_vector_unf - x_vector) / sigma_x_unf, 2)))

    unf_pdf_sigma = (x_vector_unf - x_vector) / sigma_x_unf
    return x_vector_unf, sigma_x_unf, V_x_est, V_y, unf_pdf_sigma


def svd_unfolding(signal, detector_response_matrix, cutoff=None):
    u, s, v = np.linalg.svd(detector_response_matrix, full_matrices=True)

    if signal.ndim == 2:
        sum_signal_per_chamber = np.sum(signal, axis=1)
        signal = np.histogram(sum_signal_per_chamber, bins=detector_response_matrix.shape[0])[0]

    singular_values = np.diag(s)
    new_s = np.zeros_like(singular_values)
    if cutoff:
        for j, value in singular_values:
            if j < cutoff:
                new_s[j] = singular_values[j]
            else:
                new_s[j] = 0.0
        s = np.diag(new_s)

    # So the USV*x = USV*true_energy = signal
    # So need to undo that
    d = np.dot(u.T, signal)

    # d_i = s_iz_i so z_i = d_i/s_i
    z_i = np.zeros_like(s)
    for index, i, in enumerate(d):
        z_i[index] = d[index] / s[index]

    # Now do it with V to get the unfolded distrubtion
    unfolded_signal = np.dot(z_i, v)

    # And so x = Vz, but only if you know true_energy beforehand
    '''
    # Here we are rescaling the unknowns and redefining the response matrix
    # First step is the multiply each column of Aij by the true distribution x(ini)j
    # Think this does that
    rescaled_response_matrix = detector_response_matrix * true_energy

    # Second step is define new unknowns w_j = xj/x(ini)j
    # Gives the deviation of x from the initial MC input vector
    w_j = signal / true_energy

    # Third step is to rescale the equations to have error os +-1 always.
    # In uncorreleated errors, achieved by dividing each row of Aij as well as bi by the error delta(bi)

    # TODO: Figure out what the delta(bi) error is, not sure how to get it right now

    # Now solve the rescaled system
    # Not sure whether the sigma_j Aij*wj = bi means we have to sum over A before continuing or later
    # Or something else

    rescaled_u, rescaled_s, rescaled_v = np.linalg.svd(rescaled_response_matrix, full_matrices=True)

    rescaled_z = np.dot(rescaled_v.T, w_j)
    rescaled_z = rescaled_z #* true_energy
    try:
        assert np.isclose(np.sum(rescaled_z - true_energy), 0.)
    except AssertionError:
        print("--------------------------\nRescaled Equation is not correct, does not invert back to self\n")
        print(rescaled_z)
    rescaled_d = np.dot(rescaled_u.T, signal)

    rescaled_z_i = rescaled_d / rescaled_s
    rescaled_unfolded = np.dot(rescaled_v, rescaled_z_i)
    # From paper, to get back correctly nrmalized unfolded solution have to multiply unfolded w by xini. true_energy
    rescaled_unfolded = rescaled_unfolded * true_energy

    print("Differences (Rescaled):")
    print(rescaled_unfolded - true_energy)
    '''
    # Error propagation

    return unfolded_signal, d, s, z_i


def llh_unfolding(signal, true_energy, detector_response_matrix, tau, unfolding=True, plot=False, num_bins=20):
    # If we need the Hessian, the Numdifftools should give it to us with this

    # Pretty sure should only need the response matrix Hessian, since that gives the curvature of the probabilities
    # that a given energy is in the correct bucket, so using the gradient descent, descend down the probability curvature
    # to get the most likely true distribution based off the measured values.
    # Not sure what log-likliehood does with it, maybe easier to deal the the probabilities?

    hessian_detector = nd.Hessian(detector_response_matrix)

    def likelihood(f, actual_observed, detector_matrix, tau):
        print(f)
        print(actual_observed)
        part_one = []
        for index, gi in enumerate(actual_observed):
            part_one.append(((np.sum(detector_matrix[:, index]) * f) ** actual_observed))
        part_one = np.asarray(part_one)
        print(part_one.shape)
        print(part_one)
        part_two = []
        for gi in actual_observed:
            part_two.append(np.math.factorial(gi))
        part_two = np.asarray(part_two)
        print(part_two.shape)
        print(part_two)
        part_three = np.exp(-1. * f)
        print(part_three.shape)
        print(part_three)
        result = np.prod(part_one / part_two * part_three)
        print(result)
        return result

    def calculate_C(data):
        diagonal = np.zeros_like(data)
        diagonal[0, 0] = -1
        diagonal[0, 1] = 1
        diagonal[-1, -2] = 1
        # diagonal[-1,0] = -1
        diagonal[-1, -1] = -1
        # diagonal[0,-1] = -1
        for i in range(diagonal.shape[0])[1:-1]:
            diagonal[i, i] = -2
            diagonal[i, i + 1] = 1
            diagonal[i, i - 1] = 1
        return diagonal

    def log_likelihood(f, actual_observed, detector_matrix, tau, C, regularized=True):
        print("Shape f:" + str(f.shape))
        print("Shape detector_side: " + str(detector_matrix.shape))
        print("Observed shape:" + str(actual_observed.shape))
        before_regularize = []
        for i in range(len(actual_observed)):
            # Not sure if this is correct change from the product one to this, if the summing over the ith column in A is what is supposed to be
            # the case, or the ith row, or something else
            # Mathy, it should be Sum(ln(gi!) - gi*ln(f(x)) - fi(x) * the rest
            # Part One = ln(g_i!)
            part_one = 0  # math.log(np.math.factorial(actual_observed[i]))
            # Part Two = gi*ln((Af(x)_i)
            part_two = actual_observed[i] * np.log(np.dot(detector_matrix, f)[i])
            # print("Shape part two: " + str(part_two.shape))
            # Part Three = (Af(x)_i)
            # TODO Check this ordering of i, is correct, also in Part Two
            part_three = (np.dot(detector_matrix, f)[i])
            # print("Shape part three: " + str(part_three.shape))
            # print("Shape of three parts: " + str((part_one - part_two + part_three).shape))
            before_regularize.append(part_one - part_two + part_three)
        # Prior is the 1/2 * tau * f(x).T * C' * f(x)
        if regularized:
            prior = (0.5 * tau * f.T * C.T * np.identity(C.shape[0]) * C * f)
        else:
            prior = 0
        # print(prior)
        print(len(before_regularize))
        likelihood_log = np.sum((np.asarray(before_regularize)) - prior)  # + np.diag(prior)
        print(likelihood_log)
        # print(max(likelihood_log))
        # But what happens to the 1/ root(2pi^n *det (tau 1)) part? Just disappears?
        return likelihood_log

    def gradient(f, actual_observed, detector_matrix, tau, C_prime, regularized=True):
        # Have to calculate dS/df_k = h_k = below? K is an index, so gradient is an array with k fixed per run through i
        inside_gradient = np.zeros_like(detector_matrix)
        h = np.ndarray(shape=f.shape)
        for k in range(f.shape[0]):
            for i in range(detector_matrix.shape[0]):
                part_one = detector_matrix[i, k]
                part_two = (actual_observed[i] * detector_matrix[i, k])
                part_three = np.sum((detector_matrix[i, :] * f))
                # print("Sum of Ai,j and f: " + str(part_three))
                inside_gradient[k] = (part_one - part_two / part_three)
            # I think this adds it too many times
            if regularized:
                prior = tau * np.sum(f * C_prime[k, :])
            else:
                prior = 0
            h[k] = np.sum(inside_gradient[k]) - prior
        return h

    def hessian_matrix(f, actual_observed, detector_matrix, tau, C_prime, regularized=True):
        H = np.ndarray(shape=detector_matrix.shape)
        # Trying to get d^2S/df_kdf_l = Hk,l = This?
        for k in range(f.shape[0]):
            for l in range(f.shape[0]):
                for i in range(f.shape[0]):
                    top_part = actual_observed[i] * detector_matrix[i, k] * detector_matrix[i, l]
                    bottom_part = np.sum(detector_matrix[i, :] * f) ** 2
                    if regularized:
                        prior = tau * C_prime[k, l]
                    else:
                        prior = 0
                    H[k, l] = np.sum(top_part / bottom_part) - prior

        # Now the variance/error is the inverse of the Hessian, so might as well compute it here
        error = np.linalg.inv(H)
        return H, error

    def delta_a(hessian, gradient):
        # From the paper, Delta_a = -H^-1*h, with h = gradient for the unregularized setup
        return np.dot((-1. * np.linalg.inv(hessian)), gradient)

    def iterate_unfolding(f, actual_observed, detector_matrix, tau, C, delta_a, gradient, hessian, regularized=True):
        part_one = log_likelihood(delta_a, actual_observed, detector_matrix, tau, C, regularized=regularized)
        part_two = np.dot((f - delta_a).T, gradient)
        part_three = 0.5 * np.dot(np.dot((f - delta_a).T, hessian), (f - delta_a))
        new_true = part_one + part_two + part_three
        C_prime = C.T * np.identity(C.shape[0]) * C
        new_gradient = gradient(delta_a, actual_observed, detector_matrix, tau, C_prime, regularized=regularized)
        new_hessian = hessian_matrix(delta_a, actual_observed, detector_matrix, tau, C_prime, regularized=regularized)
        new_delta_a = delta_a(new_hessian, new_gradient)

        return new_true, new_gradient, new_hessian, new_delta_a

        # Now to actually try to do the likelihood part

    # First need to bin the true energy, same as the detector matrix currently

    if signal.ndim == 2:
        sum_signal_per_chamber = np.sum(signal, axis=1)
        signal = np.histogram(sum_signal_per_chamber, bins=detector_response_matrix.shape[0])[0]
    if true_energy.ndim == 2:
        sum_signal_per_chamber = np.sum(true_energy, axis=1)
        true_energy = np.histogram(sum_signal_per_chamber, bins=detector_response_matrix.shape[0])[0]
    else:
        true_energy = np.histogram(true_energy, bins=detector_response_matrix.shape[0])[0]

    C = calculate_C(np.diag(signal))
    C_prime = C.T * np.identity(C.shape[0]) * C
    # not_log_like = likelihood(true_energy, signal, detector_response_matrix, tau)
    # plt.plot(not_log_like)
    # plt.show()
    likelihood_value = log_likelihood(true_energy, signal, detector_response_matrix, tau=tau, C=C)
    better_likelihood = log_likelihood(true_energy, true_energy, np.identity(true_energy.shape[0]), tau, C)
    print("Difference between Perfect Detector - Non Perfect one: " + str(better_likelihood - likelihood_value))
    #    fig = plt.figure()
    #    xval = np.arange(0.0, likelihood_value.shape[0])
    if plot:
        plt.plot(likelihood_value)
        plt.title("Log-Likelihood")
        plt.xlabel("Bin Number")
        plt.ylabel("Likelihood")
        # plt.yscale('log')
        plt.show()
    gradient = gradient(true_energy, signal, detector_response_matrix, tau=tau, C_prime=C_prime)
    print("Gradient Shape: " + str(gradient.shape))
    print("Gradient: " + str(gradient))
    hessian = hessian_matrix(true_energy, signal, detector_response_matrix, tau=tau, C_prime=C_prime)
    delt_a = delta_a(hessian, gradient)
    print("Delta a shape: " + str(delt_a.shape))
    pprint("Delta a: " + str(delt_a))
    print("Hessian Shape: " + str(hessian[0].shape))
    print("Hessian Diagonal: " + str(np.diag(hessian[0])))
    #    print("Difftools Hessian Diagonal: " + str(np.diag(hessian_detector)))
    # Now that likelihood is determined, either go with forward folding or unfolding
    if unfolding:
        # Now need gradient descent to find the most likely value
        change_in_a = delta_a(hessian, gradient)
        new_true = 0
        while change_in_a.all() < 5:
            new_true, new_gradient, new_hessian, change_in_a = iterate_unfolding(true_energy, signal,
                                                                                 detector_response_matrix,
                                                                                 tau=tau,
                                                                                 C=C,
                                                                                 delta_a=change_in_a,
                                                                                 gradient=gradient,
                                                                                 hessian=hessian,
                                                                                 regularized=True)
            print(change_in_a)
        print("Difference between real true and new true (Real True - New True): " + str(true_energy - new_true))
        return
    else:
        # Forward folding occurs, using Wilks Theorem to fit curve to data
        wilks_bins = 4

        return
