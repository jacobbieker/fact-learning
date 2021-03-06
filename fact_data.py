import matplotlib

matplotlib.use('Agg')

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
                      theta2_cut=0.0, conf_cut=0.9, num_off_positions=1, analysis_type='classic', with_runs=False):
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

    if with_runs:
        runs = read_h5py(sourcefile, key='runs')
        t_obs = runs.ontime.sum()

    n_events_per_off_region = len(off_data) / num_off_positions
    n_events_on_region = len(on_data)
    n_events_expected_signal = n_events_on_region - n_events_per_off_region

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
    # compare_binning_svd(digitized_classic, binned_E_validate, "Classic")
    # compare_binning_svd(digitized_closest, binned_E_validate, "Closest")
    # compare_binning_svd(digitized_lowest, binned_E_validate, "Lowest")

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


    def generate_acceptance_correction(vec_f_truth,
                                       binning,
                                       logged_truth):
        e_min = 200
        e_max = 50000
        gamma = -2.7
        n_showers = 12000000
        if logged_truth:
            binning = np.power(10., binning)
        normalization = (gamma + 1) / (e_max ** (gamma + 1) - e_min ** (gamma + 1))
        corsika_cdf = lambda E: normalization * E ** (gamma + 1) / (gamma + 1)
        vec_acceptance = np.zeros_like(vec_f_truth, dtype=float)
        for i, vec_i_detected in enumerate(vec_f_truth):
            p_bin_i = corsika_cdf(binning[i + 1]) - corsika_cdf(binning[i])
            vec_acceptance[i] = p_bin_i * n_showers / vec_i_detected
        flux_factor = 1  # 1 / (27000.**2 * np.pi * measurement['t_obs'])
        return vec_acceptance * flux_factor


    '''
    binned_e corresponds to binned_E_validate, both are binnings of true_energy 
    '''

    vec_acceptance = generate_acceptance_correction(
        vec_f_truth=vec_f,
        binning=binning_energy,
        logged_truth=False,
    )

    print(vec_acceptance)

    # Get the full energy and all that from teh gustav_werner
    gustav_gamma = pd.read_hdf("gamma_gustav_werner_corsika.hdf5", key="table")
    true_total_energy = gustav_gamma.get("energy").values

    binning_energy = np.linspace(min(true_total_energy)-1e-3, max(true_total_energy)+1e-3, real_bins)

    # binned_true_validate = tree_binning.digitize(true_total_energy)
    binned_E_true_validate = np.digitize(true_total_energy, binning_energy)

    true_counted_bin = np.bincount(binned_E_true_validate)
    true_counted_bin = true_counted_bin[1:]
    detector_true_counted_bin = np.bincount(binned_E_validate)

    print("Detector True Shape: " + str(detector_matrix_tree.shape))
    print("Vec Acceptance Shape: " + str(vec_acceptance.shape))
    print("Real Bins Value: " + str(real_bins))
    print("True Counted Value: " + str(true_counted_bin.shape))
    print("Detector True Counted Bin: " + str(detector_true_counted_bin))
    print("Real Count:" + str(true_counted_bin))
    print("Binned E Validate Max:" + str(max(binned_E_validate)))
    print("Binned True Validate Max: " + str(max(binned_E_true_validate)))

    #true_detector = np.histogram2d(true_counted_bin, detector_true_counted_bin)[0]
    #true_detector2 = np.histogram2d(true_counted_bin, true_counted_bin)[0]

    # Generated number / Total number selected in the bin = acceptance function
    # So have the true number, generated number is the binned_g_validate
    # Total real in teh detector is binned_E_energy
    # So should just do true_counted_bin / detector_true_counted_bin

    # vec_f includes only those selected events that really hit the detector
    # So even if wrong, vec_acceptance should be the same value
    # Instead it is between 1000 and 10 million times larger

    acceptance_vector_true = true_counted_bin / vec_f

    acceptance_difference = acceptance_vector_true / vec_acceptance

    print("Acceptance Difference (True / Calculated): ")
    print(acceptance_difference)

    # Plot the difference in the conversion back to true

    true_acceptance_graphing_points = acceptance_vector_true * vec_f
    calc_acceptance_graphing_points = vec_acceptance * vec_f
    x_acceptance_graphing_bins = np.linspace(0,real_bins-1, real_bins-1)
    plt.clf()
    plt.step(x_acceptance_graphing_bins, true_acceptance_graphing_points, where="mid", label="True Acceptance")
    #plt.step(x_acceptance_graphing_bins, true_counted_bin, where="mid", label="Raw True Acceptance")
    plt.step(x_acceptance_graphing_bins, calc_acceptance_graphing_points, where="mid", label="Calc Acceptance")
    plt.legend(loc='best')
    plt.yscale("log")
    #plt.show()
    plt.savefig("Acceptance_Functions_all.png")

    # A = np.histogram2d(x=)

    # Get the bin counts and then add them in the histogram

    # linear_binning_model = ff.model.LinearModel()

    # true_total_detector = linear_binning_model.A

    # Now can subtract from each to get the correctance factor, see if its the same as the vec_acceptance


    #other_acceptance_vec = 0  # Get this from counting the raw counts of the truth vs the others

    mcmc_fact_results = unfolding.mcmc_unfolding(vec_g, vec_f, detector_matrix_tree, num_walkers=100, num_threads=1, num_used_steps=100,
                                                 num_burn_steps=100, random_state=1347)
    #evaluate_unfolding.plot_corner(mcmc_fact_results[0], energies=binned_E_validate, title="TreeBinning_4000")
