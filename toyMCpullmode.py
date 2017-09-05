import matplotlib
matplotlib.use('Agg')
import funfolding as ff
import detector
import evaluate_unfolding
import numpy as np
import matplotlib.pyplot as plt

def SVD_Unf(model, vec_y, vec_x):
    A = model.A
    m = A.shape[0]
    n = A.shape[1]
    U, S_values, V_T = np.linalg.svd(A)
    order = np.argsort(S_values)[::-1]
    S_inv = np.zeros((n, m))
    for i in order:
        S_inv[i, i] = 1. / np.real(S_values[i])
    A_inv = np.dot(V_T.T, np.dot(S_inv, U.T))

    vec_x_est = np.dot(A_inv, vec_y)
    vec_x_est = np.real(vec_x_est)
    V_y = np.diag(vec_y)
    V_x_est = np.real(np.dot(A_inv, np.dot(V_y, A_inv.T)))
    factor = np.sum(vec_y) / np.sum(vec_x_est)
    vec_x_est *= factor
    V_x_est *= factor

    vec_b = np.dot(V_T, vec_x)
    vec_b_est = np.dot(V_T, vec_x_est)

    V_b = np.dot(V_T.T, np.dot(V_x_est, V_T))
    sigma_b = np.sqrt(np.diag(V_b))

    return vec_x_est, V_x_est, vec_b, sigma_b, vec_b_est, S_values[order]

def generate_data(random_state=None, noise=True, smearing=True, resolution_val=1., noise_val=0., response_bins=20,
                  rectangular_bins=20):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    energies = 1000.0 * random_state.power(0.70, 50000)
    below_zero = energies < 1.0
    energies[below_zero] = 1.0

    detector_thing = detector.Detector(distribution='gaussian',
                                       energy_loss='const',
                                       make_noise=noise,
                                       smearing=smearing,
                                       resolution_chamber=resolution_val,
                                       noise=noise_val,
                                       response_bins=response_bins,
                                       rectangular_bins=rectangular_bins,
                                       random_state=random_state)

    return detector_thing.simulate(energies)

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

for i in range(500):
    dataset = generate_data(i, response_bins=11, rectangular_bins=11)
    reloaded_data = dataset

    print(reloaded_data[0].shape)
    binned_g = reloaded_data[0]
    binned_f = reloaded_data[1]

    mmodel = ff.model.LinearModel()
    mmodel.initialize(digitized_obs=binned_g,
                      digitized_truth=binned_f)

    vec_y, vec_x = mmodel.generate_vectors(
        digitized_obs=binned_g,
        digitized_truth=binned_f)

    vec_x_est, V_x_est, vec_b, sigma_b, vec_b_est, s_values = SVD_Unf(
        mmodel, vec_y, vec_x)

    svd = ff.solution.SVDSolution()

    svd.initialize(model=mmodel, vec_g=vec_y)
    vec_f_est, V_f_est = svd.fit()
    str_1 = ''
    for f_i_est, f_i in zip(vec_f_est, vec_b):
        str_1 += '{0:.5f}\t'.format(f_i_est / f_i)
    print('{}'.format(str_1))

    normed_b = np.absolute(vec_b / sigma_b)
    normed_b_est = np.absolute(vec_b_est / sigma_b)
    order = np.argsort(normed_b)[::-1]

    normed_b = normed_b[order]
    normed_b_est = normed_b_est[order]
    binning = np.linspace(0, len(normed_b), len(normed_b))

    plt.step(binning, vec_x, where="mid", label="True Energy")
    plt.step(binning, vec_f_est, where="mid", label="SVD Unfolding")
    #plt.errorbar(x=binning, y=vec_f_est, yerr=V_f_est, fmt=".")
    plt.xlabel("Bin Number")
    plt.ylabel("log(Count)")
    plt.title("ToyMC SVD Unfolding")
    plt.legend(loc='best')
    plt.savefig("output/toyMC_svd_unfolding_" + str(i) + ".png")
    plt.clf()

    u, eigenvalues, v = np.linalg.svd(mmodel.A)
    evaluate_unfolding.plot_eigenvalues(eigenvalues)
    plt.clf()

    eigenvalues, eigenvectors = np.linalg.eig(mmodel.A)
    true, folded, measured = obtain_coefficients(np.bincount(binned_g), np.bincount(binned_f), eigenvalues, eigenvectors,)
    evaluate_unfolding.plot_eigenvalue_coefficients(true, folded, measured, 1)
    plt.clf()

    n_dims = 10

    eigen_vals, eigen_vecs = np.linalg.eig(mmodel.A)
    eigen_vals = np.absolute(eigen_vals)
    sorting = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[sorting]
    kappa = max(eigen_vals)/min(eigen_vals)
    inv_eigen_vals = 1/eigen_vals
    highest_amp = np.max(np.absolute(inv_eigen_vals))

    U = eigenvectors[:, sorting]
    f, axes = plt.subplots(U.shape[1], sharex=True, sharey=True, figsize=(10, 4*U.shape[1]))
    for i, ax_i in enumerate(axes):
        ax_i.bar(range(n_dims), U[:, i], color='C1', label='Eigenvector %d' % i)
        #ax_i.set_axis_off()
        ax_i.set_title('Shape Eigenvectors {}'.format(i))
        ax_i.set_xlabel('Index $j$')
    #plt.yscale('log')
    plt.savefig("output/eigenvectors_plotted_toyMC_" + str(i) + ".png")