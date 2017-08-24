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
        tree_binning = discretization.TreeBinningSklearn(random_state=random_state, min_samples_leaf=max_bins)
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
    classic_binning = discretization.ClassicBinning(
        bins = [15, 25],
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
        is_pos = dataset[: ,0] * dataset[:, 1] > 0
        y_clf[is_pos] = np.random.normal(loc=y_mean, size=np.sum(is_pos))
        y_clf[~is_pos] = np.random.normal(loc=-y_mean, size=np.sum(~is_pos))
        y_clf = np.array(y_clf >= 0, dtype=int)
        y_means = np.sqrt(dataset[:, 0]**2 + dataset[:, 0]**2)
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
    df_test = on_data[10000:]
    df_train = on_data[:10000]

    # Split into 20 /80 mix for tree/detector matrix sets
    df_detector = df_train[int(0.8*len(df_train)):]
    df_tree = df_train[:int(0.8*len(df_train))]

    real_bins=21
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
    tree_dataset = (df_tree.corsika_evt_header_total_energy, df_tree.gamma_energy_prediction)
    closest_binned, lowest_binned, closest_binning, lowest_binning = try_different_classic_binning(df_tree, real_bins, 25, similar_test=True)
    tree_repr, digitized_tree = decision_tree(df_tree, df_tree, tree_obs=tree_obs, max_bins=100)
    counts_per_bin = np.zeros(shape=(len(np.unique(digitized_tree))))
    for element in digitized_tree:
        counts_per_bin[element] += 1
    closest_bins, X_vals = classic_tree(df_test, df_train)

    real_energy = df_tree.corsika_evt_header_total_energy
    binning_f = np.linspace(min(real_energy) - 1e-3, max(real_energy) + 1e-3, real_bins)
    binned_f = np.digitize(real_energy, binning_f)
    binned_true = np.histogram(binned_f, bins=binning_f)[0]
    real_energy_binned = np.zeros(binning_f.shape)
    for element in binned_f:
        real_energy_binned[element] += 1
    real_energy_binned = real_energy_binned[1:]

    classic_binning = discretization.ClassicBinning(
        bins = [real_bins, signal_bins],
    )

    raw_points = np.asarray([df_tree.corsika_evt_header_total_energy, df_tree.gamma_energy_prediction])
    classic_binning.fit(raw_points)

    classic_binned = classic_binning.histogram(raw_points)
    classic_binned_array = np.outer(classic_binned, real_energy_binned)

    if False:
        threshold = 100

        closest_binned = classic_binning.merge(raw_points,
                                                min_samples=threshold,
                                                max_bins=None,
                                                mode='closest')

        closest_binned = closest_binned.histogram(raw_points)
        lowest_binned = classic_binning.merge(raw_points,
                                               min_samples=threshold,
                                               max_bins=None,
                                               mode='lowest')
        lowest_binned = lowest_binned.histogram(raw_points)

    closest_binned_array = np.outer(closest_binned, real_energy_binned)
    lowest_binned_array = np.outer(lowest_binned, real_energy_binned)
    tree_binned_array = np.outer(counts_per_bin, real_energy_binned)
    similar_reg_binned_array = np.outer(closest_binning, real_energy_binned)
    similar_binned_array = np.outer(lowest_binning, real_energy_binned)

    # See if this normalization helps at all...
    def normalizer(response_matrix):
        normalizer = np.diag(1. / np.sum(response_matrix, axis=0))
        response_matrix = np.dot(response_matrix, normalizer)
        return response_matrix
    closest_binned_array = normalizer(closest_binned_array)
    lowest_binned_array = normalizer(lowest_binned_array)
    tree_binned_array = normalizer(tree_binned_array)
    classic_binning_array = normalizer(classic_binned_array)
    similar_reg_binned_array = normalizer(similar_reg_binned_array)
    similar_binned_array = normalizer(similar_binned_array)

    u, c_s, v = np.linalg.svd(closest_binned_array)
    u, l_s, v = np.linalg.svd(lowest_binned_array)
    u, tree_s, v = np.linalg.svd(tree_binned_array)
    u, classic_s, v = np.linalg.svd(classic_binned_array)
    u, s_c, v = np.linalg.svd(similar_reg_binned_array)
    s, s_l, v = np.linalg.svd(similar_binned_array)

    # Do one over max to get condition numbers
    c_s = c_s/max(c_s)
    l_s = l_s/max(l_s)
    s_c = s_c/max(s_c)
    s_l = s_l/max(s_l)
    tree_s = tree_s/max(tree_s)
    classic_s = classic_s/max(classic_s)

    # Plot the step function of the numbers
    step_function_x_c = np.linspace(0, c_s.shape[0], c_s.shape[0])
    step_function_x_l = np.linspace(0, l_s.shape[0], l_s.shape[0])
    step_function_x_t = np.linspace(0, tree_s.shape[0], tree_s.shape[0])
    step_function_x_class = np.linspace(0, classic_s.shape[0], classic_s.shape[0])
    step_function_x_c_s = np.linspace(0, s_c.shape[0], s_c.shape[0])
    step_function_x_l_s = np.linspace(0, s_l.shape[0], s_l.shape[0])

    plt.step(step_function_x_c, c_s, where="mid", label="Closest Binning (k: " + str(1.0/min(c_s)))
    plt.step(step_function_x_l, l_s, where="mid", label="Lowest Binning (k: " + str(1.0/min(l_s)))
    plt.step(step_function_x_c_s, s_c, where="mid", label="Similar Reg Binned (k: " + str(1.0/min(s_c)))
    plt.step(step_function_x_l_s, s_l, where="mid", label="Similar Binned (k: " + str(1.0/min(l_s)))
    plt.step(step_function_x_t, tree_s, where="mid", label="Tree Binning (k: " + str(1.0/min(tree_s)))
    plt.step(step_function_x_class, classic_s, where="mid", label="Classic Binning (k: " + str(1.0/min(classic_s)))
    plt.xlabel("Singular Value Number")
    plt.legend(loc="best")
    plt.ylim(0, 1.5e-16)
    plt.show()

    closest_binning_two = np.linspace(0, closest_binned.shape[0]+1, closest_binned.shape[0])

    # Need to fix the error with real_energy and closest_binned not having same number of elements (either bin real_energy or unbin closest_binned?

    # Now go through the different binnings to get response matrix and singular values
    #get_response_matrix2(binned_true, lowest_binned, lowest_binned.shape[0], binning_f)
    #get_response_matrix2(binned_true, counts_per_bin, counts_per_bin.shape[0], binning_f)

    # Now have the counts per each bin in the fit
    if False:
        plt.plot(counts_per_bin)
        plt.title("Counts per bin from TreeBinningSklearn")
        plt.ylabel("Counts")
        plt.xlabel("Bin Number")
        plt.show()

    # Get the eigenvalues/vectors of the results to compare vs the noise
    real_energy = df_detector.corsika_evt_header_total_energy
    # Detected energy is, I think, the gamma_energy_prediction
    detected_energy = df_detector.gamma_energy_prediction

    # Try plotting like the V. Blobel Paper

    binning_f = np.linspace(min(real_energy) - 1e-3, max(real_energy) + 1e-3, real_bins)
    binning_g = np.linspace(min(detected_energy) - 1e-3, max(detected_energy) + 1e-3, signal_bins)

    binned_g = np.digitize(detected_energy, binning_g)
    binned_f = np.digitize(real_energy, binning_f)

    binned_signal = np.histogram(binned_g, bins=binning_g)[0]
    binned_true = np.histogram(binned_f, bins=binning_f)[0]

    # Gets the detector response matrix based on the 80 percent of the training data
    detector_matrix = get_response_matrix2(real_energy, detected_energy, binning_g, binning_f)
    # get_response_matrix2(real_energy, closest_binned, closest_binning_two, binning_f)

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

    model = ff.model.BasicLinearModel()
    model.initialize(g=binned_g,
                     f=binned_f)

    vec_g, vec_f = model.generate_vectors(binned_g, binned_f)

    # Get the V. Blobel plot for the measured distribution



    # Plot different binning's singular values







