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


def decision_tree(dataset_test, dataset_train, tree_obs, max_bins=None, random_state=None):
    X_tree = dataset_train.get(tree_obs).values
    X_tree_test = dataset_test.get(tree_obs).values
    binning_E = np.linspace(2.4, 4.2, 10)
    binned_E = np.digitize(dataset_train.corsika_evt_header_total_energy,
                           binning_E)
    binned_E_test = np.digitize(dataset_test.corsika_evt_header_total_energy,
                                binning_E)
    if max_bins is None:
        tree_binning = discretization.TreeBinningSklearn(random_state=random_state)
    else:
        tree_binning = discretization.TreeBinningSklearn(random_state=random_state, max_leaf_nodes=max_bins)
    tree_binning.fit(X_tree_test, binned_E_test)

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
    classic_binning = ff.discretization.ClassicBinning(
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
                                    min_samples=10,
                                    max_bins=20,
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

    # response_matrix.shape[1] is the one for f, the true distribution binning
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
    df_test = on_data[10000:]
    df_train = on_data[:10000]

    real_bins=16
    signal_bins=26

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

    tree_repr, digitized_tree = decision_tree(df_test, df_train, tree_obs=tree_obs, max_bins=25)
    closest_bins, X_vals = classic_tree(df_test, df_train)

    binned_closest = closest_bins.histogram(X_vals)

    print(len(np.unique(digitized_tree)))
    print(len(digitized_tree))

    counts_per_bin = np.zeros(shape=(len(np.unique(digitized_tree))))
    for element in digitized_tree:
        counts_per_bin[element] += 1

    # Now have the counts per each bin in the fit
    if False:
        plt.plot(counts_per_bin)
        plt.title("Counts per bin from TreeBinningSklearn")
        plt.ylabel("Counts")
        plt.xlabel("Bin Number")
        plt.show()

        evaluate_unfolding.plot_unfolded_vs_true(counts_per_bin, energies=binned_closest,
                                                 title="Counts from TreeBinningSklrean")

    # Get the eigenvalues/vectors of the results to compare vs the noise
    real_energy = df_test.corsika_evt_header_total_energy
    # Detected energy is, I think, the gamma_energy_prediction
    detected_energy = df_test.gamma_energy_prediction

    # Try plotting like the V. Blobel Paper

    binning_f = np.linspace(min(real_energy) - 1e-3, max(real_energy) + 1e-3, real_bins)
    binning_g = np.linspace(min(detected_energy) - 1e-3, max(detected_energy) + 1e-3, signal_bins)

    binned_g = np.digitize(detected_energy, binning_g)
    binned_f = np.digitize(real_energy, binning_f)

    binned_signal = np.histogram(binned_g, bins=binning_g)[0]
    binned_true = np.histogram(binned_f, bins=binning_f)[0]

    detector_matrix = get_response_matrix2(real_energy, detected_energy, binning_g, binning_f)

    model = ff.model.BasicLinearModel()
    model.initialize(g=binned_g,
                     f=binned_f)
    #plt.clf()
    #plt.imshow(model.A)
    #plt.savefig('02_matrix_A.png')
    print('\nNormalized Matrix saved as: 02_matrix_A.png')

    vec_g, vec_f = model.generate_vectors(binned_g, binned_f)

    # Get the V. Blobel plot for the measured distribution
    u, s, v = np.linalg.svd(detector_matrix, full_matrices=False)
    measured_coeffs = abs(np.dot(u.T, vec_g))
    x_steps_two = np.linspace(0,len(measured_coeffs), len(measured_coeffs))
    plt.step(x_steps_two, measured_coeffs, where="mid")
    plt.savefig("output/vec_g_dot_u.T_measured.png")
    plt.clf()

    true_coeffs = abs(np.dot(vec_f, u.T))
    x_steps_one = np.linspace(0,len(true_coeffs), len(true_coeffs))
    plt.step(x_steps_one, true_coeffs, where="mid")
    plt.savefig("output/vec_f_dot_u.T_measured.png")
    plt.clf()





    print(vec_f)
    svd = ff.solution.SVDSolution()
    print('\n===========================\nResults for each Bin: Unfolded/True')

    print('\nSVD Solution for diffrent number of kept singular values:')
    for i in range(1, detector_matrix.shape[1]):
        vec_f_est, V_f_est = svd.run(vec_g=vec_g,
                                     model=model,
                                     keep_n_sig_values=i)
        #svd.evaluate_singular_values(vec_g=vec_g,
        #                             model=model)
        #plt.show()
        #print("Shape of estimate, min, max: " + str(V_f_est.shape) + " " + str(min(V_f_est.all())) + " " + str(max(V_f_est.all())))
        str_0 = '{} singular values:'.format(str(i).zfill(2))
        str_1 = ''
        for f_i_est, f_i in zip(vec_f_est, vec_f):
            str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
        print('{}\t{}'.format(str_0, str_1))
        max_error = []
        for element in V_f_est:
            max_error.append(np.mean(element))
        x_steps = np.linspace(0, detector_matrix.shape[1]+1, detector_matrix.shape[1])
        # Plot with Equidistant Binning
        plt.clf()
        plt.title("Equidistant Binning")
        plt.errorbar(x=x_steps, y=vec_f_est, yerr=max_error, fmt=".")
        plt.hist(x_steps, weights=vec_f_est, bins=detector_matrix.shape[1], histtype='step', normed=False, label="SVD Unfolding bin counts")
        plt.hist(x_steps, weights=vec_f, bins=detector_matrix.shape[1], histtype='step', normed=False, label='True Distribution')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.savefig("output/" + str_0 + "_self_vec_g.png")
        plt.clf()

        # Plot with Tree Binning
        model = ff.model.BasicLinearModel()
        model.initialize(g=digitized_tree,
                         f=binned_f)
        print('\n===========================\nResults for each Bin: Unfolded/True')
        vec_g, vec_f = model.generate_vectors(digitized_tree, binned_f)
        print('\nSVD Solution for diffrent number of kept sigular values:')
        for j in range(1, detector_matrix.shape[1]):
            vec_f_est_tree, V_f_est_tree = svd.run(vec_g=counts_per_bin,
                                         model=model,
                                         keep_n_sig_values=j)
            unfolded_coeffs = abs(np.dot(vec_f_est_tree, u.T))
            x_steps_one = np.linspace(0,len(unfolded_coeffs), len(unfolded_coeffs))
            unfolded_not_tree_coeffs = abs(np.dot(vec_f_est, u.T))
            plt.step(x_steps_one, unfolded_coeffs, where="mid")
            plt.savefig("output/singular_vals_" + str(i) + "_vec_f_est_tree_dot_u.T_measured.png")
            plt.clf()
            plt.step(x_steps_one, unfolded_coeffs, where="mid", label="Unfolded Coeffs")
            plt.step(x_steps_one, true_coeffs, where="mid", label="True Coeffs")
            plt.legend(loc="best")
            plt.savefig("output/sin_vals_" + str(i) + "_true_and_unfolded_coeffs.png")
            plt.clf()
            plt.step(x_steps_one, unfolded_coeffs, where="mid", label="Unfolded Coeffs")
            plt.step(x_steps_one, true_coeffs, where="mid", label="True Coeffs")
            plt.step(x_steps_two, measured_coeffs, where="mid", label="Measured Coeffs")
            plt.legend(loc="best")
            plt.savefig("output/sin_vals_" + str(i) + "_true_measured_unfolded_coeffs.png")
            plt.clf()
            plt.step(x_steps_one, unfolded_coeffs, where="mid", label="Unfolded Coeffs")
            plt.step(x_steps_one, true_coeffs, where="mid", label="True Coeffs")
            plt.step(x_steps_two, measured_coeffs, where="mid", label="Measured Coeffs")
            plt.step(x_steps_one, unfolded_not_tree_coeffs, where="mid", label="Classic Binning Unfolded Coeffs")
            plt.legend(loc="best")
            plt.savefig("output/sin_vals_" + str(i) + "_true_measured_both_unfolded_coeffs.png")
            plt.clf()
            #print("Shape of estimate, min, max: " + str(V_f_est.shape) + " " + str(min(V_f_est.all())) + " " + str(max(V_f_est.all())))
            str_0 = '{} singular values:'.format(str(j).zfill(2))
            str_1 = ''
            for f_i_est, f_i in zip(vec_f_est_tree, vec_f):
                str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
            print('{}\t{}'.format(str_0, str_1))
            max_error = []
            for element in V_f_est_tree:
                max_error.append(np.mean(element))
            x_steps = np.linspace(0, detector_matrix.shape[1]+1, detector_matrix.shape[1])
            plt.clf()
            plt.title("Tree Learn Binning")
            plt.errorbar(x=x_steps, y=vec_f_est_tree, yerr=max_error, fmt=".")
            plt.hist(x_steps, weights=vec_f_est_tree, bins=detector_matrix.shape[1], histtype='step', normed=False, label="SVD Unfolding bin counts")
            plt.hist(x_steps, weights=vec_f, bins=detector_matrix.shape[1], histtype='step', normed=False, label='True Distribution')
            plt.yscale('log')
            plt.legend(loc='best')
            plt.savefig("output/" + str_0 + "_self_tree_bin.png")

            # Plot both
            plt.clf()
            plt.title("Tree Learn Binning vs Equidistant Binning")
            plt.errorbar(x=x_steps, y=vec_f_est_tree, yerr=max_error, fmt=".", alpha=0.4)
            plt.errorbar(x=x_steps, y=vec_f_est, yerr=max_error, fmt=".", alpha=0.4)
            plt.hist(x_steps, weights=vec_f_est_tree, bins=detector_matrix.shape[1], histtype='step', normed=False, label="SVD Unfolding Tree bin counts")
            plt.hist(x_steps, weights=vec_f_est, bins=detector_matrix.shape[1], histtype='step', normed=False, label="SVD Unfolding bin counts")
            plt.hist(x_steps, weights=vec_f, bins=detector_matrix.shape[1], histtype='step', normed=False, label='True Distribution')
            plt.yscale('log')
            plt.legend(loc='best')
            plt.savefig("output/" + str_0 + "_self_tree_bin_vec_g_compare.png")

        get_eigenvalues_and_condition(detector_matrix, full_matrix=True)







