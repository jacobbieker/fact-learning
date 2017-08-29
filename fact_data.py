import funfolding as ff
from funfolding import discretization
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import corner
import evaluate_unfolding
import unfolding

import multiprocessing

from fact.io import read_h5py
from fact.analysis import split_on_off_source_independent
from fact.analysis import split_on_off_source_dependent

import logging
from configparser import ConfigParser

'''
Parameters for the classic fitting of the gamma data, taken from the TRUEE settings
# 1st observable
branch_y: gamma_energy_prediction
number_y_bins: 25 log # just an example for logarithmic input
# limits_y: 0 0

branch_y: conc_core
number_y_bins: 15 log
# limits_y: 0 0


branch_y: zd_tracking
number_y_bins: 3
# limits_y: {min_zenith} {max_zenith}
'''

'''
TODO: Run 500 or so iterations, with 10,000 MC data points set aside as the test, and the rest used to train on, changing the testing set each time

Make the unfolding vs raw with noise level plot for the output

Try with classic equidistant binning and then the non equidistant one

Use SVD unfolding first since it is fast to run

Run on vollmond probably, since it takes a long time


Create a thing to enable giving data and running multiple tests with different binnings and unfolding ways at once

Create plot of the singular value / max(singular value) for the different binnings or SVD unfoldings....And get condition

Add support for using the a vector  to get unfolded back to true distribution (f * a elementwise) in the regularization

Use 20 percent for training tree, 80 percent for detector matrix

For the tree, use lots of bins, set a min number of elements per leaf, not a set number of bins for better results

Error for SVD is covariance matrix, just use diagonal for now for the errors [X]

So use the digitized g and f vectors to create the detector response by zip(vec_f, vec_g): 
A[i,j] += 1 

or something similar. Almsot right but not quite at the moment. Should have two digistized ones that are the lengths of the input number of particles and then just slot them into the detector matrix. Tree has a thing that can do that, looke at testing.py. 

Have to change linspace in testing since thats for icecube data, same with the energy things

Basically need digitized things to make it work correctly and to build the detector matricies, which is where we get the singular values from.

'''

log = logging.getLogger("setup_pypet")

# df = pd.read_hdf("gamma_precuts.hdf5")
# print(list(df))
print("+++++++++++++++++++++++++++++++++++++++++++")


# mc_df = read_h5py("gamma_test.hdf5", key='events')
# print(list(mc_df))


def load_gamma_subset(sourcefile,
                      theta2_cut=0.0, conf_cut=0.9, num_off_positions=1, analysis_type='classic'):
    events = read_h5py(sourcefile, key='events')

    selection_columns = ['theta_deg', 'gamma_prediction', 'zd_tracking', 'conc_core']
    theta_off_columns = ['theta_deg_off_{}'.format(i)
                         for i in range(1, num_off_positions)]
    bg_prediction_columns = ['gamma_prediction_off_{}'.format(i)
                             for i in range(1, num_off_positions)]

    if analysis_type == 'source':
        log.info('\tSelection events for source dependent analysis')
        log.info("\t\tgamma_pred_cut={0:.2f}".format(conf_cut))
        on_data, off_data = split_on_off_source_dependent(
            events=events,
            prediction_threshold=conf_cut,
            on_prediction_key='gamma_prediction',
            off_prediction_keys=bg_prediction_columns)
        on_mc = events.query('gamma_prediction >= {}'.format(conf_cut))
    elif analysis_type == 'classic':
        log.info('\tSelection events for source independent analysis')
        log.info("\t\tgamma_pred_cut={0:.2f}".format(conf_cut))
        log.info("\t\ttheta2_cut={0:.2f}".format(theta2_cut))
        on_data, off_data = split_on_off_source_independent(
            events=events.query('gamma_prediction >= {}'.format(conf_cut)),
            theta2_cut=theta2_cut,
            theta_key='theta_deg',
            theta_off_keys=theta_off_columns)
        on_mc = events.query(
            '(theta_deg <= {}) & (gamma_prediction >= {})'.format(
                theta2_cut, conf_cut))

    log.info("\t{} Data Events (on region)".format(len(on_data)))
    log.info("\t\t{} Data Events ({} off regions)".format(len(off_data),
                                                          num_off_positions))
    log.info("\t{} MC gammas after selection".format(len(on_mc)))

    return on_mc, on_data, off_data


def convert_to_log(dataset):
    dataset.corsika_evt_header_total_energy = np.log10(
        dataset.corsika_evt_header_total_energy)
    dataset.gamma_energy_prediction = np.log10(dataset.gamma_energy_prediction)
    # dataset.conc_core= np.log10(dataset.conc_core)
    dataset.size = np.log10(dataset.size)
    dataset.length = np.log10(dataset.length)
    dataset.num_pixel_in_shower = np.log10(
        dataset.num_pixel_in_shower)
    return dataset


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


def decision_tree(dataset_test, dataset_train, tree_obs, max_bins=None, random_state=None):
    X_tree = dataset_train.get(tree_obs).values
    X_tree_test = dataset_test.get(tree_obs).values
    binning_E = np.linspace(2.4, 4.2, 10)
    binned_E = np.digitize(dataset_train.corsika_evt_header_total_energy,
                           binning_E)
    binned_E_test = np.digitize(dataset_test.corsika_evt_header_total_energy,
                                binning_E)
    if max_bins is None:
        tree_binning = ff.binning.TreeBinningSklearn(random_state=random_state)
    else:
        tree_binning = ff.binning.TreeBinningSklearn(random_state=random_state, min_samples_leaf=max_bins)
    tree_binning.fit(X_tree_test, binned_E_test)

    print("Number of bins: " + str(tree_binning.n_bins))

    print(tree_binning.tree.tree_.feature)
    print(len(tree_binning.tree.tree_.feature))

    # tree_binning = discretization.TreeBinning()

    # tree_binning.fit(X_tree_test, binned_E_test)

    # print(tree_binning.tree.feature)
    # print(tree_binning.tree.threshold)
    return tree_binning, tree_binning.digitize(X_tree_test)


def classic_tree(dataset_test, dataset_train, plot=False):
    X = dataset_train.get(['conc_core', 'gamma_energy_prediction']).values
    X_test = dataset_test.get(['conc_core', 'gamma_energy_prediction']).values

    binning_E = np.linspace(2.4, 4.2, 10)
    binned_E = np.digitize(dataset_train.corsika_evt_header_total_energy,
                           binning_E)
    binned_E_test = np.digitize(dataset_test.corsika_evt_header_total_energy,
                                binning_E)
    classic_binning = ff.binning.ClassicBinning(
        bins=[15, 25])
    classic_binning.fit(X)
    if plot:
        fig, ax = plt.subplots()
        ff.discretization.visualize_classic_binning(ax,
                                                    classic_binning,
                                                    X,
                                                    log_c=True,
                                                    cmap='viridis')
        fig.savefig('05_fact_example_original_binning_log.png')

    closest = classic_binning.merge(X_test,
                                    min_samples=100,
                                    max_bins=None,
                                    mode='closest')
    if plot:
        fig, ax = plt.subplots()
        ff.discretization.visualize_classic_binning(ax,
                                                    closest,
                                                    X,
                                                    log_c=True,
                                                    cmap='viridis')

        fig.savefig('05_fact_example_original_binning_closest_log.png')

    return closest, X


def try_different_classic_binning(dataset, bins_true=15, bins_measured=25, similar_test=False):
    classic_binning = ff.binning.ClassicBinning(
        bins=[15, 25],
    )
    dataset = dataset.values
    for index, mini_array in enumerate(dataset):
        mini_array[0] = min(mini_array[1:])
        mini_array = np.float64(mini_array)
        dataset[index] = mini_array
    dataset = np.float64(dataset)
    print(dataset.shape)
    classic_binning.fit(dataset)

    threshold = 100

    closest = classic_binning.merge(dataset,
                                    min_samples=threshold,
                                    max_bins=None,
                                    mode='closest')
    binned_closest = closest.histogram(dataset)

    lowest = classic_binning.merge(dataset,
                                   min_samples=threshold,
                                   max_bins=None,
                                   mode='lowest')
    binned_lowest = lowest.histogram(dataset)
    if similar_test:
        y_mean = 1.5
        y_clf = np.zeros(dataset.shape[0])
        is_pos = dataset[:, 0] * dataset[:, 1] > 0
        y_clf[is_pos] = np.random.normal(loc=y_mean, size=np.sum(is_pos))
        y_clf[~is_pos] = np.random.normal(loc=-y_mean, size=np.sum(~is_pos))
        y_clf = np.array(y_clf >= 0, dtype=int)
        y_means = np.sqrt(dataset[:, 0] ** 2 + dataset[:, 0] ** 2)
        y_reg = np.random.normal(loc=y_means,
                                 scale=0.3)
        similar_clf = classic_binning.merge(dataset,
                                            min_samples=threshold,
                                            max_bins=None,
                                            mode='similar',
                                            y=y_clf)
        binned_similar_clf = similar_clf.histogram(dataset)

        similar_reg = classic_binning.merge(dataset,
                                            min_samples=threshold,
                                            max_bins=None,
                                            mode='similar',
                                            y=y_reg)
        binned_similar_reg = similar_reg.histogram(dataset)

    if similar_test:
        return binned_closest, binned_lowest, binned_similar_clf, binned_similar_reg
    else:
        return binned_closest, binned_lowest, closest, lowest


def get_binning(original_energy_distribution, signal):
    binnings = []
    for r in [(min(signal), max(signal)), (min(original_energy_distribution), max(original_energy_distribution))]:
        low = r[0]
        high = r[-1]
        binnings.append(np.linspace(low, high + 1, high - low + 2))
    return binnings[0], binnings[1]


def get_response_matrix2(original_energy_distribution, signal, binning_g, binning_f):
    response_matrix = np.histogram2d(signal, original_energy_distribution, bins=(binning_g, binning_f))[0]
    normalizer = np.diag(1. / np.sum(response_matrix, axis=0))
    response_matrix = np.dot(response_matrix, normalizer)

    # response_matrix.shape[1] is the one for f, /.the true distribution binning
    # response_matrix.shape[0] is for the binning of g
    print(response_matrix.shape)

    return response_matrix


def get_eigenvalues_and_condition(detector_matrix, full_matrix=False):
    u, s, v = np.linalg.svd(detector_matrix, full_matrices=full_matrix)
    eigen_vals = np.absolute(s)
    sorting = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[sorting]
    # U = eigen_vecs
    D = np.diag(eigen_vals)
    kappa = max(eigen_vals) / min(eigen_vals)
    print("Kappa:\n", str(kappa))
    return eigen_vals, D, kappa


if __name__ == '__main__':
    mc_data, on_data, off_data = load_gamma_subset("gamma_test.hdf5", theta2_cut=0.7, conf_cut=0.3, num_off_positions=5)

    # 0's are off
    on_data = convert_to_log(on_data)

    print(on_data.shape)
    print(off_data.shape)
    print(mc_data.shape)

    # Get the "test" vs non test data
    df_train = on_data[10000:]
    df_test = on_data[:10000]

    # Split into 20 /80 mix for tree/detector matrix sets
    df_detector = df_train[int(0.8 * len(df_train)):]
    df_tree = df_train[:int(0.8 * len(df_train))]

    real_bins = 21
    signal_bins = 26

    tree_obs = ["size",
                "width",
                "length",
                "m3_trans",
                "m3_long",
                "conc_core",
                "m3l",
                "m3t",
                "concentration_one_pixel",
                "concentration_two_pixel",
                "leakage",
                "leakage2",
                "conc_cog",
                "num_islands",
                "num_pixel_in_shower",
                "ph_charge_shower_mean",
                "ph_charge_shower_variance",
                "ph_charge_shower_max"]

    # Now try it the other way of making the detector response with the digitized values


    detected_energy_test = df_test.get(tree_obs).values
    real_energy_test = df_test.get("corsika_evt_header_total_energy").values

    detected_energy_tree = df_tree.get(tree_obs).values
    real_energy_tree = df_tree.get("corsika_evt_header_total_energy").values

    detected_energy_detector = df_detector.get(tree_obs).values
    real_energy_detector = df_detector.get("corsika_evt_header_total_energy").values

    binning_energy = np.linspace(2.3, 4.7, real_bins)
    binning_detected = np.linspace(2.3, 4.7, signal_bins)

    binned_E_validate = np.digitize(real_energy_detector, binning_energy)
    binned_E_train = np.digitize(real_energy_tree, binning_detected)

    threshold = 100

    tree_binning = ff.binning.TreeBinningSklearn(
        regression=False,
        max_features=None,
        min_samples_split=2,
        max_depth=None,
        min_samples_leaf=threshold,
        max_leaf_nodes=None,
        random_state=1337)

    tree_binning.fit(detected_energy_tree, binned_E_train)
    binned_g_validate = tree_binning.digitize(detected_energy_detector)

    linear_binning_model = ff.model.LinearModel()
    linear_binning_model.initialize(
        digitized_obs=binned_g_validate,
        digitized_truth=binned_E_validate
    )

    detector_matrix_tree = linear_binning_model.A

    # Now have the tree binning response matrix, need classic binning ones

    classic_binning = ff.binning.ClassicBinning(
        bins=[real_bins, signal_bins],
    )
    # testing_dataset = np.asarray([detected_energy_tree, real_energy_tree])
    classic_binning.initialize(detected_energy_tree)

    digitized_classic = classic_binning.digitize(detected_energy_detector)

    linear_binning_model.initialize(
        digitized_obs=digitized_classic,
        digitized_truth=binned_E_validate
    )

    detector_matrix_classic = linear_binning_model.A

    closest = classic_binning.merge(detected_energy_detector,
                                    min_samples=threshold,
                                    max_bins=None,
                                    mode='closest')
    digitized_closest = closest.digitize(detected_energy_detector)

    lowest = classic_binning.merge(detected_energy_detector,
                                   min_samples=threshold,
                                   max_bins=None,
                                   mode='lowest')
    digitized_lowest = lowest.digitize(detected_energy_detector)

    linear_binning_model.initialize(
        digitized_obs=digitized_closest,
        digitized_truth=binned_E_validate
    )
    detector_matrix_closest = linear_binning_model.A

    linear_binning_model.initialize(
        digitized_obs=digitized_lowest,
        digitized_truth=binned_E_validate
    )
    detector_matrix_lowest = linear_binning_model.A

    u, tree_singular_values, v = np.linalg.svd(detector_matrix_tree)
    u, closest_singular_values, v = np.linalg.svd(detector_matrix_closest)
    u, lowest_singular_values, v = np.linalg.svd(detector_matrix_lowest)
    u, classic_singular_values, v = np.linalg.svd(detector_matrix_classic)

    tree_singular_values = tree_singular_values / max(tree_singular_values)
    closest_singular_values = closest_singular_values / max(closest_singular_values)
    lowest_singular_values = lowest_singular_values / max(lowest_singular_values)
    classic_singular_values = classic_singular_values / max(classic_singular_values)

    step_function_x_c = np.linspace(0, closest_singular_values.shape[0], closest_singular_values.shape[0])
    step_function_x_l = np.linspace(0, lowest_singular_values.shape[0], lowest_singular_values.shape[0])
    step_function_x_t = np.linspace(0, tree_singular_values.shape[0], tree_singular_values.shape[0])
    step_function_x_class = np.linspace(0, classic_singular_values.shape[0], classic_singular_values.shape[0])

    plt.step(step_function_x_c, closest_singular_values, where="mid",
             label="Closest Binning (k: " + str(1.0 / min(closest_singular_values)))
    plt.step(step_function_x_l, lowest_singular_values, where="mid",
             label="Lowest Binning (k: " + str(1.0 / min(lowest_singular_values)))
    plt.step(step_function_x_t, tree_singular_values, where="mid",
             label="Tree Binning (k: " + str(1.0 / min(tree_singular_values)))
    plt.step(step_function_x_class, classic_singular_values, where="mid",
             label="Classic Binning (k: " + str(1.0 / min(classic_singular_values)))
    plt.xlabel("Singular Value Number")
    plt.legend(loc="best")
    plt.yscale('log')
    plt.savefig("Singular_Values.png")
    plt.clf()

    # Blobel Thing
    fig, ax = plt.subplots()
    def compare_binning_svd(binned_g, binned_E, title):
        linear_binning_model = ff.model.LinearModel()

        # binned_g_validate = binned_g_validate[:10000]
        # binned_E_validate = binned_E_validate[:10000]
        linear_binning_model.initialize(digitized_obs=binned_g,
                                        digitized_truth=binned_E)

        vec_y, vec_x = linear_binning_model.generate_vectors(
            digitized_obs=binned_g,
            digitized_truth=binned_E)

        vec_x_est, V_x_est, vec_b, sigma_b, vec_b_est, s_values = SVD_Unf(
            linear_binning_model, vec_y, vec_x)

        svd = ff.solution.SVDSolution()

        svd.initialize(model=linear_binning_model, vec_g=vec_y)
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
        binning = np.linspace(0, len(normed_b), len(normed_b) + 1)
        bin_centers = (binning[1:] + binning[:-1]) / 2
        bin_width = (binning[1:] - binning[:-1]) / 2

        ax.hist(bin_centers, bins=binning, weights=normed_b_est, label='Unfolded: ' + title,
                histtype='step')

    compare_binning_svd(binned_g_validate, binned_E_validate, "Tree")
    #compare_binning_svd(digitized_classic, binned_E_validate, "Classic")
    #compare_binning_svd(digitized_closest, binned_E_validate, "Closest")
    #compare_binning_svd(digitized_lowest, binned_E_validate, "Lowest")

    linear_binning_model.initialize(digitized_obs=binned_g_validate,
                                    digitized_truth=binned_E_validate)
    # binned_E_validate += 1
    vec_g, vec_f = linear_binning_model.generate_vectors(binned_g_validate, binned_E_validate)

    # Get the V. Blobel plot for the measured distribution

    vec_y, vec_x = linear_binning_model.generate_vectors(
        digitized_obs=binned_g_validate,
        digitized_truth=binned_E_validate)

    vec_x_est, V_x_est, vec_b, sigma_b, vec_b_est, s_values = SVD_Unf(
        linear_binning_model, vec_y, vec_x)

    normed_b = np.absolute(vec_b / sigma_b)
    normed_b_est = np.absolute(vec_b_est / sigma_b)
    order = np.argsort(normed_b)[::-1]

    normed_b = normed_b[order]
    normed_b_est = normed_b_est[order]
    binning = np.linspace(0, len(normed_b), len(normed_b) + 1)
    bin_centers = (binning[1:] + binning[:-1]) / 2
    bin_width = (binning[1:] - binning[:-1]) / 2

    ax.hist(bin_centers,
            bins=binning,
            weights=normed_b,
            label='Truth',
            histtype='step')


    ax.axhline(1.)
    ax.set_xlabel(r'Index $j$')
    ax.set_ylabel(r'$\left|b_j/\sigma_j\right|$')
    ax.set_ylim([1e-2, 1e3])
    ax.set_yscale("log", nonposy='clip')
    ax.legend(loc='best')
    fig.savefig('08_classic_binning.png')

    mcmc_fact_results = unfolding.mcmc_unfolding(vec_g, vec_f, detector_matrix_tree, num_threads=8, num_used_steps=2000, random_state=1347)
    evaluate_unfolding.plot_corner(mcmc_fact_results[0], energies=binned_E_validate, title="TreeBinning_4000")

