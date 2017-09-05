import matplotlib

matplotlib.use('Agg')

import funfolding as ff
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import corner
import evaluate_unfolding

from fact.io import read_h5py
from fact.analysis import split_on_off_source_independent
from fact.analysis import split_on_off_source_dependent


# df = pd.read_hdf("gamma_precuts.hdf5")
# if plot:
# print(list(df))
if plot:
    print("+++++++++++++++++++++++++++++++++++++++++++")


# mc_df = read_h5py("gamma_test.hdf5", key='events')
# if plot:
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
        on_data, off_data = split_on_off_source_dependent(
            events=events,
            prediction_threshold=conf_cut,
            on_prediction_key='gamma_prediction',
            off_prediction_keys=bg_prediction_columns)
        on_mc = events.query('gamma_prediction >= {}'.format(conf_cut))
    elif analysis_type == 'classic':
        on_data, off_data = split_on_off_source_independent(
            events=events.query('gamma_prediction >= {}'.format(conf_cut)),
            theta2_cut=theta2_cut,
            theta_key='theta_deg',
            theta_off_keys=theta_off_columns)
        on_mc = events.query(
            '(theta_deg <= {}) & (gamma_prediction >= {})'.format(
                theta2_cut, conf_cut))

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


def split_mc_test_unfolding(n_pulls,
                            n_events_mc,
                            n_events_test,
                            n_events_A=-1,
                            n_events_binning='n_events_test',
                            random_state=None):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    if isinstance(n_events_test, float):
        if n_events_test > 0 and n_events_test < 1:
            n_events_test = n_events_mc * n_events_test
        else:
            n_events_test = int(n_events_test)
    elif not isinstance(n_events_test, int):
        raise ValueError("'n_events_test' must be either None, int or float")

    if n_events_A is None:
        n_events_A = None
    elif isinstance(n_events_A, float):
        n_events_A = int(n_events_mc * n_events_A)
    elif not isinstance(n_events_A, int):
        raise ValueError("'n_events_A' must be either None, int or "
                         "float")

    if n_events_binning is None:
        n_events_binning = 0
    elif isinstance(n_events_binning, float):
        n_events_binning = int(n_events_mc * n_events_binning)
    elif isinstance(n_events_binning, str):
        if n_events_binning.lower() == 'n_events_test':
            n_events_binning = int(n_events_test)
        else:
            raise ValueError(
                "'{}'' unknown option for 'n_events_binning'".format(
                    n_events_binning))
    else:
        raise ValueError("'n_events_binning' must be either None, int or "
                         "float")

    if (n_events_test + n_events_binning + n_events_A) > n_events_mc:
        raise ValueError("'n_events_test' + 'n_events_binning' + 'n_events_A' "
                         "has to be smaller than n_events_mc")
    n_events_test_pulls = random_state.poisson(n_events_test,
                                            size=n_pulls)
    idx = np.arange(n_events_mc)

    for n_events_test_i in n_events_test_pulls:
        random_state.shuffle(idx)
        test_idx = np.sort(idx[:n_events_test_i])
        train_idx = idx[n_events_test_i:]

        if n_events_binning == 0:
            if n_events_A == -1:
                A_idx = np.sort(train_idx)
            else:
                A_slice = slice(None, n_events_A)
                A_idx = np.sort(idx[A_slice])
            yield test_idx, A_idx
        else:
            binning_slice = slice(None, n_events_binning)
            binning_idx = np.sort(idx[binning_slice])
            if n_events_A == -1:
                A_slice = slice(n_events_binning, None)
            else:
                A_slice = slice(n_events_binning,
                                n_events_binning + n_events_A)
            A_idx = np.sort(idx[A_slice])
            yield test_idx, A_idx, binning_idx


if __name__ == '__main__':
    mc_data, on_data, off_data = load_gamma_subset("gamma_test.hdf5", theta2_cut=0.7, conf_cut=0.3, num_off_positions=5)

    # 0's are off
    on_data = convert_to_log(on_data)
    gustav_gamma = pd.read_hdf("gamma_gustav_werner_corsika.hdf5", key="table")
    # Get the full energy and all that from teh gustav_werner
    true_total_energy = np.log10(gustav_gamma.get("energy").values)

    if plot:
        print(on_data.shape)
    if plot:
        print(off_data.shape)
    if plot:
        print(mc_data.shape)

    # Just to save some Memory
    del off_data
    del mc_data

    # Now call this 500 times to get the variance, etc... Change plotting as well
    number_pulls = 500
    num_events_mc = on_data.shape[0]-1
    num_events_test = 10000
    num_events_A = 0.1
    random_state = 1347

    testing_data = split_mc_test_unfolding(number_pulls, num_events_mc, num_events_test, num_events_A, random_state=random_state)

    # Generator so go through it calling the events each time

    list_of_mcmc_errors = []
    list_of_acceptance_errors = []
    list_of_tree_condition_numbers = []
    list_of_classic_conditions = []
    list_of_closest_conditions = []
    list_of_lowest_conditions = []

    run = 0
    plot = False
    for indicies in testing_data:
        run += 1
        if plot:
            print(run)

        # Get the "test" vs non test data
        df_test = on_data.loc[indicies[0]].dropna(how='all')
        df_train = on_data[~on_data.isin(df_test)].dropna(how='all')

        # Split into 20 /80 mix for tree/detector matrix sets
        df_tree = df_train[int(0.8 * len(df_train)):]
        df_detector = df_train[:int(0.8 * len(df_train))]

        real_bins = 10
        signal_bins = 16

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

        binned_E_test_validate = np.digitize(real_energy_test, binning_energy)

        threshold = 1000

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
        binned_g_test = tree_binning.digitize(detected_energy_test)

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

        list_of_tree_condition_numbers.append(1.0 / min(tree_singular_values))
        list_of_classic_conditions.append(1.0 / min(classic_singular_values))
        list_of_closest_conditions.append(1.0 / min(closest_singular_values))
        list_of_lowest_conditions.append(1.0 / min(lowest_singular_values))
        if plot:
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
            plt.savefig("output/Singular_Values_" + str(run) + ".png")
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
            if plot:
                print('{}'.format(str_1))
            if plot:
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
        if plot:
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
            fig.savefig('output/08_classic_binning_m+testing_' + str(run) + '.png')
        plt.clf()

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

        vec_acceptance = generate_acceptance_correction(
            vec_f_truth=vec_f,
            binning=binning_energy,
            logged_truth=True,
        )

        binned_E_true_validate = np.digitize(true_total_energy, binning_energy)

        true_counted_bin = np.histogram(true_total_energy, bins=binning_energy)[0]
        detector_true_counted_bin = np.bincount(binned_E_validate)

        acceptance_vector_true = true_counted_bin / vec_f

        acceptance_difference = acceptance_vector_true / vec_acceptance

        if plot:
            print(acceptance_difference)
        list_of_acceptance_errors.append(acceptance_difference)

        # Plot the difference in the conversion back to true

        true_acceptance_graphing_points = acceptance_vector_true * vec_f
        calc_acceptance_graphing_points = vec_acceptance * vec_f
        x_acceptance_graphing_bins = np.linspace(0,real_bins-1, real_bins-1)
        plt.clf()

        def test_different_binnings(observed_energy, true_energy, title, tau=None, acceptance_vector=None, log_f=True):
            model = ff.model.LinearModel()
            model.initialize(digitized_obs=observed_energy,
                             digitized_truth=true_energy)

            vec_g, vec_f = model.generate_vectors(observed_energy, true_energy)

            if plot:
                print('\nMCMC Solution: (constrained: sum(vec_f) == sum(vec_g)) : (FIRST RUN)')

            llh = ff.solution.StandardLLH(tau=tau,
                                          vec_acceptance=acceptance_vector,
                                          C='thikonov',
                                          log_f=log_f,
                                          neg_llh=False)
            llh.initialize(vec_g=vec_g,
                           model=model)

            sol_mcmc = ff.solution.LLHSolutionMCMC(n_used_steps=8000,
                                                   random_state=1337)
            sol_mcmc.initialize(llh=llh, model=model)
            sol_mcmc.set_x0_and_bounds()
            vec_f_est_mcmc, sigma_vec_f, samples, probs = sol_mcmc.fit()
            str_0 = 'unregularized:'
            str_1 = ''
            for f_i_est, f_i in zip(vec_f_est_mcmc, vec_f):
                str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
            if plot:
                print('{}\t{}'.format(str_0, str_1))

            list_of_mcmc_errors.append([vec_f_est_mcmc, sigma_vec_f, vec_f])
            # Every third one is of the same type

            if plot:
                plt.clf()
                evaluate_unfolding.plot_unfolded_vs_true(vec_f_est_mcmc, vec_f, sigma_vec_f, title=str(title + "_" + str(run)))
                plt.close()

            if plot:
                print('\nMinimize Solution:')
            llh = ff.solution.StandardLLH(tau=None,
                                          C='thikonov',
                                          neg_llh=True)
            llh.initialize(vec_g=vec_g,
                           model=model)

            sol_mini = ff.solution.LLHSolutionMinimizer()
            sol_mini.initialize(llh=llh, model=model)
            sol_mini.set_x0_and_bounds()

            solution, V_f_est = sol_mini.fit(constrain_N=False)
            vec_f_est_mini = solution.x
            str_0 = 'unregularized:'
            str_1 = ''
            for f_i_est, f_i in zip(vec_f_est_mini, vec_f):
                str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
            if plot:
                print('{}\t{}'.format(str_0, str_1))

            if plot:
                print('\nMinimize Solution (constrained: sum(vec_f) == sum(vec_g)):')
            solution, V_f_est = sol_mini.fit(constrain_N=True)
            vec_f_est_mini = solution.x
            str_0 = 'unregularized:'
            str_1 = ''
            for f_i_est, f_i in zip(vec_f_est_mini, vec_f):
                str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
            if plot:
                print('{}\t{}'.format(str_0, str_1))

            if plot:
                print('\nMinimize Solution (MCMC as seed):')
            sol_mini.set_x0_and_bounds(x0=vec_f_est_mcmc)
            solution, V_f_est = sol_mini.fit(constrain_N=False)
            vec_f_est_mini = solution.x
            str_0 = 'unregularized:'
            str_1 = ''
            for f_i_est, f_i in zip(vec_f_est_mini, vec_f):
                str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
            if plot:
                print('{}\t{}'.format(str_0, str_1))

            if plot:
                corner.corner(samples, truths=vec_f)
                plt.savefig('corner_truth' + title + '.png')
                if plot:
                    print(np.sum(vec_f_est_mcmc))

                plt.clf()
                corner.corner(samples, truths=vec_f_est_mini, truth_color='r')
                plt.savefig('corner_mini' + title + '.png')
                plt.clf()
                corner.corner(samples, truths=vec_f_est_mcmc, truth_color='springgreen')
                plt.savefig('corner_mcmc' + title + '.png')
                plt.clf()


        test_different_binnings(binned_g_test, binned_E_test_validate, "Tree Binning_" + str(run))

        # Now have the tree binning response matrix, need classic binning ones

        classic_binning = ff.binning.ClassicBinning(
            bins=[real_bins, signal_bins],
        )

        classic_binning.initialize(detected_energy_tree)

        digitized_classic = classic_binning.digitize(detected_energy_test)

        closest = classic_binning.merge(detected_energy_test,
                                        min_samples=threshold,
                                        max_bins=None,
                                        mode='closest')
        digitized_closest = closest.digitize(detected_energy_test)

        lowest = classic_binning.merge(detected_energy_test,
                                       min_samples=threshold,
                                       max_bins=None,
                                       mode='lowest')
        digitized_lowest = lowest.digitize(detected_energy_test)
        plt.clf()
        test_different_binnings(digitized_lowest, binned_E_test_validate, "Lowest Binning_"+ str(run))
        plt.clf()
        test_different_binnings(digitized_closest, binned_E_test_validate, "Closest Binning_"+ str(run))
        plt.close()
        plt.clf()

    # Now plotting the different ones for the multiple runs
    def remove_nan(inputdta):
        inputdta = np.asarray(inputdta)
        inputdta = inputdta[~np.isnan(inputdta)]
        return inputdta
    list_of_mcmc_errors = remove_nan(list_of_mcmc_errors)
    list_of_acceptance_errors = remove_nan(list_of_acceptance_errors)
    list_of_tree_condition_numbers = remove_nan(list_of_tree_condition_numbers)
    list_of_classic_conditions = remove_nan(list_of_classic_conditions)
    list_of_closest_conditions = remove_nan(list_of_closest_conditions)
    list_of_lowest_conditions = remove_nan(list_of_lowest_conditions)
    with open("overall_stats.txt", "w") as output:
        output.write("Tree Binning Condition Mean and Std.: " + str(np.mean(list_of_tree_condition_numbers)) + " " + str(np.std(list_of_tree_condition_numbers)))
        output.write("Classic Binning Condition Mean and Std.: " + str(np.mean(list_of_classic_conditions)) + " " + str(np.std(list_of_classic_conditions)))
        output.write("Closest Binning Condition Mean and Std.: " + str(np.mean(list_of_closest_conditions)) + " " + str(np.std(list_of_closest_conditions)))
        output.write("Lowest Binning Condition Mean and Std.: " + str(np.mean(list_of_lowest_conditions)) + " " + str(np.std(list_of_lowest_conditions)))

        tree_error_real = list_of_mcmc_errors[0::3]
        lowest_error_real = list_of_mcmc_errors[1::3]
        closest_error_real = list_of_mcmc_errors[2::3]

        tree_error_real_lower1 = tree_error_real[0] - tree_error_real[1][0]
        tree_error_real_upper1 = tree_error_real[1][1] - tree_error_real[0]

        output.write("Tree Binning MCMC Error Lower Mean and Std.: " + str(np.mean(tree_error_real_lower1)) + " " + str(np.std(tree_error_real_lower1)))
        output.write("Tree Binning MCMC Error Lower Mean and Std.: " + str(np.mean(tree_error_real_upper1)) + " " + str(np.std(tree_error_real_upper1)))

        # Plot the mean fitting vs the mean data for Tree, Closest, Lowest

        mean_tree_sets = [np.mean(tree_error_real[0]), np.mean(tree_error_real[1]), np.mean(tree_error_real[2])]
        mean_closest_sets = [np.mean(closest_error_real[0]), np.mean(closest_error_real[1]), np.mean(closest_error_real[2])]
        mean_lowest_sets = [np.mean(lowest_error_real[0]), np.mean(lowest_error_real[1]), np.mean(lowest_error_real[2])]

        plt.clf()
        evaluate_unfolding.plot_unfolded_vs_true(mean_tree_sets[0], mean_tree_sets[2], mean_tree_sets[1], title="Mean_Tree_Binning")
        plt.close()

        plt.clf()
        evaluate_unfolding.plot_unfolded_vs_true(mean_closest_sets[0], mean_closest_sets[2], mean_closest_sets[1], title="Mean_Closest_Binning")
        plt.close()

        plt.clf()
        evaluate_unfolding.plot_unfolded_vs_true(mean_lowest_sets[0], mean_lowest_sets[2], mean_lowest_sets[1], title="Mean_Lowest_Binning")
        plt.close()

        closest_error_real_lower = closest_error_real[0] - closest_error_real[1][0]
        closest_error_real_upper = closest_error_real[1][1] - closest_error_real[0]

        output.write("Closest Binning MCMC Error Lower Mean and Std.: " + str(np.mean(closest_error_real_lower)) + " " + str(np.std(closest_error_real_lower)))
        output.write("Closest Binning MCMC Error Lower Mean and Std.: " + str(np.mean(closest_error_real_upper)) + " " + str(np.std(closest_error_real_upper)))

        lowest_error_real_lower = lowest_error_real[0] - lowest_error_real[1][0]
        lowest_error_real_upper = lowest_error_real[1][1] - lowest_error_real[0]

        output.write("Closest Binning MCMC Error Lower Mean and Std.: " + str(np.mean(lowest_error_real_lower)) + " " + str(np.std(lowest_error_real_lower)))
        output.write("Closest Binning MCMC Error Lower Mean and Std.: " + str(np.mean(lowest_error_real_upper)) + " " + str(np.std(lowest_error_real_upper)))

        tree_raw_off = tree_error_real[0] / tree_error_real[2]
        closest_raw_off = closest_error_real[0] / closest_error_real[2]
        lowest_raw_off = lowest_error_real[0] / lowest_error_real[2]

        x_raw_off = np.linspace(0, tree_raw_off.shape[0], tree_raw_off.shape[0])
        plt.clf()
        plt.step(x_raw_off, tree_raw_off, where="mid", label="Tree Binning Difference")
        plt.step(x_raw_off, closest_raw_off, where="mid", label="Closest Binning Difference")
        plt.step(x_raw_off, lowest_raw_off, where="mid", label="Lowest Binning Difference")
        plt.legend(los='best')
        plt.title("Difference between Unfolded and True Spectrum For " + str(tree_raw_off.shape[0]) + " Runs")
        plt.xlabel("Run Number")
        plt.ylabel("log(Difference)")
        plt.yscale('log')
        plt.savefig("output/Multiple_Run_Difference.png")