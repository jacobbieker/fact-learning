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
#mc_df = read_h5py("gamma_test.hdf5", key='events')
#print(list(mc_df))


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


def decision_tree(dataset_test, dataset_train, tree_obs, random_state=None):
    X_tree = dataset_train.get(tree_obs).values
    X_tree_test = dataset_test.get(tree_obs).values
    binning_E = np.linspace(2.4, 4.2, 10)
    binned_E = np.digitize(dataset_train.corsika_evt_header_total_energy,
                           binning_E)
    binned_E_test = np.digitize(dataset_test.corsika_evt_header_total_energy,
                                binning_E)

    tree_binning = discretization.TreeBinningSklearn(max_leaf_nodes=20, random_state=random_state)
    tree_binning.fit(X_tree_test, binned_E_test)

    print(tree_binning.tree.tree_.feature)
    print(len(tree_binning.tree.tree_.feature))

    #tree_binning = discretization.TreeBinning()

    #tree_binning.fit(X_tree_test, binned_E_test)

    #print(tree_binning.tree.feature)
    #print(tree_binning.tree.threshold)
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

    tree_repr, digitized_tree = decision_tree(df_test, df_train, tree_obs=tree_obs)
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

    evaluate_unfolding.plot_unfolded_vs_true(counts_per_bin, energies=binned_closest, title="Counts from TreeBinningSklrean")
