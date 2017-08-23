import numpy as np
from scipy.stats import powerlaw
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import emcee
import corner


def bin_data(signal, true_energy, detector_response_matrix):
    if true_energy.ndim == 2:
        true_energy = np.sum(true_energy, axis=1)
    if signal.ndim == 2:
        signal = np.sum(signal, axis=1)

    binning_f = np.linspace(min(true_energy) - 1e-3, max(true_energy) + 1e-3, detector_response_matrix.shape[1]+1)
    binning_g = np.linspace(min(signal) - 1e-3, max(signal) + 1e-3, detector_response_matrix.shape[0]+1)

    #3 signal = np.digitize(np.sum(signal, axis=1), binning_g)
    #true_hits = np.digitize(energies, binning_f)

    signal = np.histogram(signal, bins=binning_g)[0]
    true_energy = np.histogram(true_energy, bins=binning_f)[0]

    return signal, true_energy


def log_likelihood(f, actual_observed, detector_matrix, tau, C, regularized=True, negative_log=True, a=None):
    """
    Calculate the log likelihood and returns -inf or inf if the input involves a negative number
    :param f:
    :param actual_observed:
    :param detector_matrix:
    :param tau:
    :param C:
    :param regularized:
    :param negative_log:
    :return:
    """
    before_regularize = []
    if np.asarray(actual_observed).any() <= 0:
        if negative_log:
            return np.inf
        else:
            return -np.inf
    for element in f:
        if element <= 0:
            if negative_log:
                return np.inf
            else:
                return -np.inf
    for i in range(len(actual_observed)):
        if np.asarray(before_regularize).any() < 0 or np.asarray(f).any() < 0:
            if negative_log:
                return np.inf
            else:
                return -np.inf
        #  - gi*ln(f(x)) - fi(x) * the rest Part One = ln(g_i!)
        part_one = 0  # math.log(np.math.factorial(actual_observed[i]))
        # Part Two = gi*ln((Af(x)_i)
        part_two = actual_observed[i] * np.log(np.dot(detector_matrix, f)[i])
        # Part Three = (Af(x)_i)
        part_three = (np.dot(detector_matrix, f)[i])
        before_regularize.append(part_one - part_two + part_three)
    # Prior is the 1/2 * tau * f(x).T * C' * f(x)
    if regularized:
        prior = (0.5 * tau * np.dot(np.dot(np.log10(f.T * a + 1), np.dot(C.T, C)), np.log10(f * a + 1)))
    else:
        prior = 0
    likelihood_log = np.sum((np.asarray(before_regularize)) + prior)  # + np.diag(prior)
    if negative_log:
        return likelihood_log
    else:
        return likelihood_log * -1


def gradient_array(f, actual_observed, detector_matrix, tau, C_prime, regularized=True, a=None):
    # Have to calculate dS/df_k = h_k = below? K is an index, so gradient is an array with k fixed per run through i
    inside_gradient = np.zeros_like(detector_matrix)
    h = np.zeros(shape=f.shape, dtype=np.float64)
    for k in range(detector_matrix.shape[1]):
        possion_part = 0
        for i in range(detector_matrix.shape[0]):
            part_one = detector_matrix[i, k]
            part_two = (actual_observed[i])  # * detector_matrix[i, k])
            part_three = np.sum((np.dot(detector_matrix[i, :], f)))
            # print("Sum of Ai,j and f: " + str(part_three))
            inside_gradient[i] = (part_one - part_two / part_three)
            possion_part += part_one - part_two / part_three
        # I think this adds it too many times
        if regularized:
            prior = tau * np.sum(np.dot(C_prime[:, k], np.log10(f * a + 1)))
        else:
            prior = 0
        # print("Compare prior - np sum vs way in other one---------------------------------------------------------")
        # print(prior - np.sum(inside_gradient[k]))
        # print(prior - possion_part)
        h[k] = prior - possion_part  # np.sum(inside_gradient[k])
    return h


def hessian_matrix(f, actual_observed, detector_matrix, tau, C_prime, regularized=True):
    H = np.zeros(shape=(detector_matrix.shape[1], detector_matrix.shape[1]), dtype=np.float64)
    # print(H.shape)
    # Trying to get d^2S/df_kdf_l = Hk,l = This?
    for k in range(H.shape[1]):
        for l in range(H.shape[1]):
            possion_part = 0
            for i in range(detector_matrix.shape[0]):
                top_part = actual_observed[i] * detector_matrix[i, k] * detector_matrix[i, l]
                bottom_part = np.sum(detector_matrix[i, :] * f)

                possion_part += top_part / bottom_part ** 2
            if regularized:
                prior = tau * C_prime[k, l]
            else:
                prior = 0
            H[k, l] = possion_part + prior

    # Now the variance/error is the inverse of the Hessian, so might as well compute it here
    # print(H)
    error = np.linalg.inv(H)
    return H, error


def calculate_C(data):
    diagonal = np.zeros((max(data.shape), max(data.shape)))
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


def delta_a(hessian, gradient):
    # print("Delta Gradient shape: "+ str(gradient.shape))
    # print("Hess matirx shape: "+ str(hessian.shape))
    # From the paper, Delta_a = -H^-1*h, with h = gradient for the unregularized setup
    return 0.0005 * -1. * np.dot((np.linalg.inv(hessian)), gradient)
    # Now to actually try to do the likelihood part


def obtain_coefficients(signal, true_energy, eigen_values, eigen_vectors, cutoff=None):
    U = eigen_vectors
    eigen_vals = np.absolute(eigen_values)
    sorting = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[sorting]
    D = np.diag(eigen_vals)

    sum_signal_per_chamber = signal  # The x value
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


def eigenvalue_cutoff(signal, true_energy, detector_matrix, cutoff=None):
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

    signal, _ = bin_data(signal, signal, detector_response_matrix)

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

    A_inv = np.dot(v.T, np.dot(s, u.T))
    vec_f = np.dot(signal, A_inv.T)
    vec_f = np.real(vec_f)
    V_y = np.diag(signal)
    V_f_est = np.real(np.dot(A_inv, np.dot(V_y, A_inv.T)))
    factor = np.sum(signal) / np.sum(vec_f)
    vec_f *= factor
    V_f_est *= factor

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


def llh_unfolding(signal, true_energy, detector_response_matrix, tau, unfolding=True, plot=False, regularized=True):
    # If we need the Hessian, the Numdifftools should give it to us with this

    # Pretty sure should only need the response matrix Hessian, since that gives the curvature of the probabilities
    # that a given energy is in the correct bucket, so using the gradient descent, descend down the probability curvature
    # to get the most likely true distribution based off the measured values.
    # Not sure what log-likliehood does with it, maybe easier to deal the the probabilities?

    # First need to bin the true energy, same as the detector matrix currently

    #signal, true_energy = bin_data(signal, true_energy, detector_response_matrix)

    C = calculate_C(np.diag(signal))

    C_prime = np.dot(np.dot(C.T, np.identity(C.shape[0])), C)
    print("C Prime  vs np.dot(C.T, C)")
    print(np.isclose(C_prime.all(), np.dot(C.T, C).all()))
    # not_log_like = likelihood(true_energy, signal, detector_response_matrix, tau)
    # plt.plot(not_log_like)
    # plt.show()
    likelihood_value = log_likelihood(true_energy, signal, detector_matrix=detector_response_matrix, tau=tau, C=C,
                                      regularized=regularized, negative_log=True)
    better_likelihood = log_likelihood(true_energy, true_energy, np.identity(true_energy.shape[0]), tau, C,
                                       regularized=regularized, negative_log=True)
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
    gradient = gradient_array(true_energy, signal, detector_response_matrix, tau=tau, C_prime=C_prime, regularized=regularized)
    print("Gradient Shape: " + str(gradient.shape))
    print("Gradient: " + str(gradient))
    hessian = hessian_matrix(true_energy, signal, detector_response_matrix, tau=tau, C_prime=C_prime, regularized=regularized)
    print("Hessian Shape: " + str(hessian[0].shape))
    print("Hessian Diagonal: " + str(np.diag(hessian[0])))
    #    print("Difftools Hessian Diagonal: " + str(np.diag(hessian_detector)))
    # Now that likelihood is determined, either go with forward folding or unfolding
    if unfolding:
        # Now need gradient descent to find the most likely value
        change_in_a = delta_a(hessian[0], gradient)
        # Set to a uniform distribution now, basic power law later, if start too close to the correct one, might be cause of iteration problems
        new_true = np.ones(shape=true_energy.shape) * np.sum(true_energy) / len(true_energy)
        gradient_update = gradient
        hessian_update = hessian[0]
        hessian_error = 0
        print(change_in_a)
        iterations = 0
        new_likelihood = likelihood_value
        while 0.005 < change_in_a.any() or change_in_a.any() < -0.005:
            print(new_true)
            print(np.sum(new_true))
            old_likelihood = new_likelihood
            print("Number of Iterations: " + str(iterations))
            part_one = log_likelihood(new_true, signal, detector_matrix=detector_response_matrix,
                                      tau=tau, C=C, regularized=regularized)
            print("Part One: " + str(part_one))
            part_two = np.dot(change_in_a.T, gradient_update)
            print("Part Two: " + str(part_two))
            print("Change in A Transposed: " + str(change_in_a.T))
            print("First Dot: " + str(np.dot(change_in_a.T, hessian_update)))
            print("Second Dot: " + str(np.dot(np.dot(change_in_a.T, hessian_update), change_in_a)))
            print("Reversed Second Dot: " + str(np.dot(np.dot(hessian_update, change_in_a.T), change_in_a)))
            part_three = 0  # 0.5 * np.dot(np.dot(change_in_a.T, hessian_update), change_in_a)
            print("Part Three: " + str(part_three))
            # This is currently the log likeliehood after the change, so it should be a single number and very negative
            new_likelihood = (part_one + part_two + part_three)
            print("New vs old: " + str(new_likelihood - old_likelihood))
            print(new_likelihood)
            # Minimization check here, if the new value is larger than the old one, going wrong way, so break the loop
            # Could try different step sizes I guess
            if new_likelihood < old_likelihood:
                new_true = new_true + change_in_a
                size_of_change = min(new_true)
                new_true += -1. * size_of_change + 0.01
                new_true = new_true / np.sum(new_true)
                new_true *= np.sum(signal)
            else:
                print("BREAKING AFTER " + str(iterations) + " ITERATIONS")
                break
            # Basically, if the new_true value, the value we are trying to minimize, then if it is larger, we should follow that and change the
            # "true" distribution accordingly in that direction by the change in a, then rerun the iteration...
            gradient_update = gradient_array(new_true, signal,
                                             detector_matrix=detector_response_matrix, tau=tau, C_prime=C_prime,
                                             regularized=regularized)
            hessian_update, hessian_error = hessian_matrix(new_true, signal, detector_matrix=detector_response_matrix,
                                                           tau=tau, C_prime=C_prime, regularized=regularized)
            change_in_a = delta_a(hessian_update, gradient_update)
            iterations += 1

            print(change_in_a)
        print("Difference between real true and new true (New True/Real True):\n " + str(new_true / true_energy))
        print(iterations)
        return new_true, signal, true_energy, hessian_error
    else:
        # Forward folding occurs, using Wilks Theorem to fit curve to data
        wilks_bins = 4
        new_true = np.ones(shape=true_energy.shape) * np.sum(true_energy) / len(true_energy)
        # new_true = signal
        bounds = []
        for i in range(true_energy.shape[0]):
            bounds.append((0, np.sum(signal)))
        cons = ({'type': 'eq', 'fun': lambda x: np.absolute(np.sum(x) - np.sum(new_true))})
        solution = minimize(fun=log_likelihood,
                            x0=new_true,
                            args=(true_energy, detector_response_matrix, tau, C, regularized),
                            bounds=bounds,
                            method='SLSQP',
                            constraints=cons,
                            options={'maxiter': 1000}
                            )
        print(solution.x)
        # print("Difference between solution and true (Solution/True):\n " + str(solution.x / true_energy))
        # print("Difference between signal and real true (Signal/Real True):\n " + str(signal / true_energy))
        # print("Difference between the two above ones (New_True Array - Signal Array):\n " + str((solution.x / true_energy) - (signal / true_energy)))
        # print("Difference between Solution and Signal (Solution/Signal): \n" + str(solution.x / signal))
        print(solution.success)
        print(solution.message)

        return solution.x, signal, true_energy


def mcmc_unfolding(signal, true_energy, detector_response_matrix, num_walkers=100, num_used_steps=2000,
                   num_burn_steps=1000, random_state=None, num_threads=1, tau=0., regularized=False):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.mtrand.RandomState(random_state)

    # signal, true_energy = bin_data(signal, true_energy, detector_response_matrix)
    C = calculate_C(detector_response_matrix)
    sampler = emcee.EnsembleSampler(nwalkers=num_walkers,
                                    dim=detector_response_matrix.shape[1],
                                    lnpostfn=log_likelihood,
                                    threads=num_threads,
                                    args=(signal, detector_response_matrix, tau, C, regularized, False))

    total_steps = num_burn_steps + num_used_steps
    uniform_start = np.ones(shape=detector_response_matrix.shape[1])
    starting_positions = np.zeros((num_walkers, detector_response_matrix.shape[1]), dtype=np.float32)
    for index, value in enumerate(uniform_start):
        starting_positions[:, index] = np.abs(random_state.poisson(np.sum(signal)/len(signal), size=num_walkers))

    new_true, samples, probabilities = sampler.run_mcmc(pos0=starting_positions,
                                                        N=total_steps,
                                                        rstate0=random_state.get_state())

    samples = sampler.chain[:, num_burn_steps:, :]
    samples = samples.reshape((-1, detector_response_matrix.shape[1]))

    probabilities = sampler.lnprobability[:, num_burn_steps:]
    probabilities = probabilities.reshape((-1))

    max_likelihood = np.argmax(probabilities)
    print(samples.shape)
    print(probabilities.shape)
    print(max_likelihood)
    print(probabilities[max_likelihood])
    return samples, probabilities, new_true, probabilities[max_likelihood]
