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

    # Just to save some Memory
    del off_data
    del mc_data
    print(on_data.shape)

    # Now call this 500 times to get the variance, etc... Change plotting as well
    number_pulls = 500
    num_events_mc = on_data.shape[0] - 1
    num_events_test = 10000
    num_events_A = 0.1
    random_state = 1347

    testing_data = split_mc_test_unfolding(number_pulls, num_events_mc, num_events_test, num_events_A,
                                           random_state=random_state)

    # Generator so go through it calling the events each time

    list_of_mcmc_errors = [[], [], [], [], [], []]
    list_true = [[], [], [], [], [], []]
    list_mcmc = [[], [], [], [], [], []]
    list_of_svd_errors = [[], [], [], [], [], []]
    list_svd = [[], [], [], [], [], []]
    list_of_min_errors = [[], [], [], [], [], []]
    list_min = [[], [], [], [], [], []]
    list_measured = [[], [], [], [], [], []]
    list_of_pdf_errors = [[], [], [], [], [], []]
    list_pdf = [[], [], [], [], [], []]
    list_pdf_error = [[], [], [], [], [], []]
    list_of_acceptance_errors = []
    list_of_tree_condition_numbers = []
    list_of_classic_conditions = []
    list_of_closest_conditions = []
    list_of_lowest_conditions = []

    run = 0
    plot = True
    for indicies in testing_data:
        run += 1
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

        binned_E_true_validate = np.digitize(true_total_energy, binning_energy)

        true_counted_bin = np.histogram(true_total_energy, bins=binning_energy)[0]
        true_pdf = true_counted_bin / np.sum(true_counted_bin)
        true_pdf *= real_energy_test.shape
        true_pdf = np.ceil(true_pdf)
        true_pdf = np.int64(true_pdf)


        binned_E_validate = np.digitize(real_energy_detector, binning_energy)
        binned_E_train = np.digitize(real_energy_tree, binning_detected)

        binned_E_test_validate = np.digitize(real_energy_test, binning_energy)

        detector_true_counted_bin = np.bincount(binned_E_validate)

        pdf_digitized = np.repeat(np.arange(true_pdf.size), true_pdf)
        diff_between_pdf_true = pdf_digitized.size - real_energy_test.size
        if diff_between_pdf_true < 0:
            pdf_digitized = np.append(pdf_digitized, np.zeros(shape=diff_between_pdf_true))
        elif diff_between_pdf_true > 0:
            indicies = np.arange(0, diff_between_pdf_true, step=1)
            pdf_digitized = np.delete(pdf_digitized, indicies)
        print(pdf_digitized.shape)
        print(binned_E_test_validate.shape)
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

        step_function_x_c = np.linspace(0, closest_singular_values.shape[0], closest_singular_values.shape[0]+1)
        step_function_x_l = np.linspace(0, lowest_singular_values.shape[0], lowest_singular_values.shape[0]+1)
        step_function_x_t = np.linspace(0, tree_singular_values.shape[0], tree_singular_values.shape[0]+1)
        step_function_x_class = np.linspace(0, classic_singular_values.shape[0], classic_singular_values.shape[0]+1)

        list_of_tree_condition_numbers.append(1.0 / min(tree_singular_values))
        list_of_classic_conditions.append(1.0 / min(classic_singular_values))
        list_of_closest_conditions.append(1.0 / min(closest_singular_values))
        list_of_lowest_conditions.append(1.0 / min(lowest_singular_values))
        bin_width_c = (step_function_x_c[1:] - step_function_x_c[:-1]) / 2.
        bin_center_c = (step_function_x_c[:-1] + step_function_x_c[1:]) / 2.
        bin_width_l = (step_function_x_l[1:] - step_function_x_l[:-1]) / 2.
        bin_center_l = (step_function_x_l[:-1] + step_function_x_l[1:]) / 2.
        bin_width_t = (step_function_x_t[1:] - step_function_x_t[:-1]) / 2.
        bin_center_t = (step_function_x_t[:-1] + step_function_x_t[1:]) / 2.
        bin_width_class = (step_function_x_class[1:] - step_function_x_class[:-1]) / 2.
        bin_center_class = (step_function_x_class[:-1] + step_function_x_class[1:]) / 2.
        if plot:
            plt.clf()
            plt.hist(bin_center_c, bins=step_function_x_c, weights=closest_singular_values, histtype='step',
                     label="Closest Binning (k: " + str(1.0 / min(closest_singular_values)))
            plt.hist(bin_center_l, bins=step_function_x_l, weights=lowest_singular_values, histtype='step',
                     label="Lowest Binning (k: " + str(1.0 / min(lowest_singular_values)))
            plt.hist(bin_center_t, bins=step_function_x_t, weights=tree_singular_values, histtype='step',
                     label="Tree Binning (k: " + str(1.0 / min(tree_singular_values)))
            plt.hist(bin_center_class, bins=step_function_x_class, weights=classic_singular_values, histtype='step',
                     label="Classic Binning (k: " + str(1.0 / min(classic_singular_values)))
            plt.xlabel("Singular Value Number")
            plt.legend(loc="best")
            plt.yscale('log')
            plt.savefig("output/Singular_Values_" + str(run) + ".png")
            plt.clf()

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

        acceptance_vector_true = true_counted_bin / vec_f

        acceptance_difference = acceptance_vector_true / vec_acceptance

        if plot:
            print(acceptance_difference)
        list_of_acceptance_errors.append(acceptance_difference)

        # Plot the difference in the conversion back to true

        true_acceptance_graphing_points = acceptance_vector_true * vec_f
        calc_acceptance_graphing_points = vec_acceptance * vec_f
        x_acceptance_graphing_bins = np.linspace(0, real_bins - 1, real_bins - 1)
        plt.clf()


        def test_different_binnings(observed_energy, true_energy, title, pdf_truth, tau=None, acceptance_vector=None,
                                    log_f=True, index=0):
            model = ff.model.LinearModel()
            model.initialize(digitized_obs=observed_energy,
                             digitized_truth=true_energy)

            vec_g, vec_f = model.generate_vectors(observed_energy, true_energy)

            pdf_model = ff.model.LinearModel()
            pdf_model.initialize(digitized_obs=observed_energy,
                                 digitized_truth=pdf_truth)

            pdf_g, pdf_f = pdf_model.generate_vectors(observed_energy, pdf_truth)

            if plot:
                print('\nMCMC Solution: (constrained: sum(vec_f) == sum(vec_g)) : (FIRST RUN)')

            llh = ff.solution.StandardLLH(tau=tau,
                                          vec_acceptance=acceptance_vector,
                                          C='thikonov',
                                          log_f=log_f,
                                          neg_llh=False)

            pdf_llh = ff.solution.StandardLLH(tau=tau,
                                              vec_acceptance=acceptance_vector,
                                              C='thikonov',
                                              log_f=log_f,
                                              neg_llh=False)

            llh.initialize(vec_g=vec_g,
                           model=model)

            pdf_llh.initialize(vec_g=pdf_g,
                              model=pdf_model)

            sol_mcmc = ff.solution.LLHSolutionMCMC(n_used_steps=4000,
                                                   n_threads=4,
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

            # Get the mean error
            difference = np.asarray(vec_f_est_mcmc) / np.asarray(vec_f)
            list_of_mcmc_errors[index].append(difference)
            list_of_mcmc_errors[index + 1].append(sigma_vec_f)
            list_true[index].append(vec_f)
            list_mcmc[index].append(vec_f_est_mcmc)
            list_measured[index].append(vec_g)

            sol_mcmc.initialize(llh=pdf_llh, model=pdf_model)
            sol_mcmc.set_x0_and_bounds()
            vec_f_est_mcmc, sigma_vec_f, samples, probs = sol_mcmc.fit()
            str_0 = 'unregularized:'
            str_1 = ''

            difference = np.asarray(pdf_f) - np.asarray(vec_f_est_mcmc)
            list_of_pdf_errors[index].append(difference)
            list_of_pdf_errors[index + 1].append(sigma_vec_f)
            list_pdf[index].append(vec_f_est_mcmc)

            svd = ff.solution.SVDSolution()
            print('\n===========================\nResults for each Bin: Unfolded/True')

            print('\nSVD Solution for diffrent number of kept sigular values:')
            svd.initialize(model=model, vec_g=vec_g)
            vec_f_est, V_f_est = svd.fit()
            for f_i_est, f_i in zip(vec_f_est, vec_f):
                str_1 += '{0:.2f}\t'.format(f_i_est / f_i)
            print('{}\t{}'.format(str_0, str_1))

            difference = np.asarray(vec_f_est) / np.asarray(vec_f)
            list_of_svd_errors[index].append(difference)
            list_of_svd_errors[index + 1].append(V_f_est)
            list_svd[index].append(vec_f_est)

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
            print('{}\t{}'.format(str_0, str_1))

            difference = np.asarray(vec_f_est_mini) / np.asarray(vec_f)
            list_of_min_errors[index].append(difference)
            list_of_min_errors[index + 1].append(V_f_est)
            list_min[index].append(vec_f_est_mcmc)

            # Every third one is of the same type

            if plot:
                plt.clf()
                evaluate_unfolding.plot_unfolded_vs_true(vec_f_est_mcmc, vec_f, sigma_vec_f,
                                                         title=str(title + "_" + str(run)))
                plt.clf()

            if plot:
                corner.corner(samples, truths=vec_f)
                plt.savefig('corner_truth' + title + '.png')
                plt.clf()
                if plot:
                    print(np.sum(vec_f_est_mcmc))

                plt.clf()
                corner.corner(samples, truths=vec_f_est_mcmc, truth_color='springgreen')
                plt.savefig('corner_mcmc' + title + '.eps', format='eps', dpi=1000)
                plt.clf()

        plt.clf()
        test_different_binnings(binned_g_test, binned_E_test_validate, "Tree Binning" + str(run), pdf_truth=pdf_digitized, index=0)

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
        test_different_binnings(digitized_lowest, binned_E_test_validate,"Lowest Binning" + str(run), pdf_truth=pdf_digitized, index=2)
        plt.clf()
        test_different_binnings(digitized_closest, binned_E_test_validate, "Closest Binning" + str(run), pdf_truth=pdf_digitized, index=4)
        plt.clf()
        if run > 2:
            break


    # Now plotting the different ones for the multiple runs
    def remove_nan(inputdta):
        inputdta = np.asarray(inputdta)
        # inputdta = inputdta[~np.isnan(inputdta)]
        return inputdta


    list_of_acceptance_errors = remove_nan(list_of_acceptance_errors)
    list_of_tree_condition_numbers = remove_nan(list_of_tree_condition_numbers)
    list_of_classic_conditions = remove_nan(list_of_classic_conditions)
    list_of_closest_conditions = remove_nan(list_of_closest_conditions)
    list_of_lowest_conditions = remove_nan(list_of_lowest_conditions)
    with open("overall_stats_all.txt", "w") as output:
        x_raw_off = np.linspace(0, list_of_tree_condition_numbers.shape[0], list_of_tree_condition_numbers.shape[0]+1)
        bin_width = (x_raw_off[1:] - x_raw_off[:-1]) / 2.
        bin_center = (x_raw_off[:-1] + x_raw_off[1:]) / 2.
        plt.clf()
        plt.hist(bin_center, bins=x_raw_off, weights=list_of_closest_conditions, histtype='step',
                 label="Closest Binning")
        plt.hist(bin_center, bins=x_raw_off, weights=list_of_lowest_conditions, histtype='step', label="Lowest Binning")
        plt.hist(bin_center, bins=x_raw_off, weights=list_of_tree_condition_numbers, histtype='step',
                 label="Tree Binning")
        plt.hist(bin_center, bins=x_raw_off, weights=list_of_classic_conditions, histtype='step',
                 label="Classic Binning")
        plt.legend(loc='best')
        plt.title("Condition Numbers For " + str(list_of_tree_condition_numbers.shape[0]) + " Runs")
        plt.xlabel("Run Number")
        plt.ylabel("log(Condition)")
        plt.yscale('log')
        plt.savefig("output/Multiple_Run_Condition.png")
        plt.clf()
        output.write(
            "Tree Binning Condition Mean and Std.: " + str(np.nanmean(list_of_tree_condition_numbers)) + " " + str(
                np.nanstd(list_of_tree_condition_numbers)) + "\n")
        output.write(
            "Classic Binning Condition Mean and Std.: " + str(np.nanmean(list_of_classic_conditions)) + " " + str(
                np.nanstd(list_of_classic_conditions)) + "\n")
        output.write(
            "Closest Binning Condition Mean and Std.: " + str(np.nanmean(list_of_closest_conditions)) + " " + str(
                np.nanstd(list_of_closest_conditions)) + "\n")
        output.write(
            "Lowest Binning Condition Mean and Std.: " + str(np.nanmean(list_of_lowest_conditions)) + " " + str(
                np.nanstd(list_of_lowest_conditions)) + "\n")

        tree_error_real = np.asarray(list_of_mcmc_errors[1])
        tree_real = np.asarray(list_of_mcmc_errors[0])
        lowest_error_real = np.asarray(list_of_mcmc_errors[3])
        lowest_real = np.asarray(list_of_mcmc_errors[2])
        closest_error_real = np.asarray(list_of_mcmc_errors[5])
        closest_real = np.asarray(list_of_mcmc_errors[4])

        tree_error_real_pdf = np.asarray(list_of_pdf_errors[1])
        tree_real_pdf = np.asarray(list_of_pdf_errors[0])
        lowest_error_real_pdf = np.asarray(list_of_pdf_errors[3])
        lowest_real_pdf = np.asarray(list_of_pdf_errors[2])
        closest_error_real_pdf = np.asarray(list_of_pdf_errors[5])
        closest_real_pdf = np.asarray(list_of_pdf_errors[4])

        list_mcmc_tree = np.asarray(list_mcmc[0])
        list_mcmc_low = np.asarray(list_mcmc[2])
        list_mcmc_close = np.asarray(list_mcmc[4])
        list_true_tree = np.asarray(list_true[0])
        list_true_low = np.asarray(list_true[2])
        list_true_close = np.asarray(list_true[4])

        # Plot the mean fitting vs the mean data for Tree, Closest, Lowest

        print(list_mcmc[0])
        print(list_mcmc_tree)
        print(tree_error_real[0])

        tree_raw_off = np.nanmean(tree_real, axis=1)
        closest_raw_off = np.nanmean(closest_real, axis=1)
        lowest_raw_off = np.nanmean(lowest_real, axis=1)

        x_raw_off = np.linspace(0, tree_raw_off.shape[0], tree_raw_off.shape[0]+1)
        bin_width = (x_raw_off[1:] - x_raw_off[:-1]) / 2.
        bin_center = (x_raw_off[:-1] + x_raw_off[1:]) / 2.
        plt.clf()
        plt.hist(bin_center, bins=x_raw_off, weights=closest_raw_off, histtype='step', label="Closest Binning")
        plt.hist(bin_center, bins=x_raw_off, weights=lowest_raw_off, histtype='step', label="Lowest Binning")
        plt.hist(bin_center, bins=x_raw_off, weights=tree_raw_off, histtype='step', label="Tree Binning")
        plt.legend(loc='best')
        plt.title("Ratio between Unfolded and True Spectrum For " + str(tree_raw_off.shape[0]) + " Runs")
        plt.xlabel("Run Number")
        plt.ylabel("log(Ratio)")
        plt.yscale('log')
        plt.savefig("output/Multiple_Run_Difference.png")
        plt.clf()

        tree_raw_off = np.nanmean(tree_real, axis=0)
        closest_raw_off = np.nanmean(closest_real, axis=0)
        lowest_raw_off = np.nanmean(lowest_real, axis=0)

        tree_raw_off_std = np.nanstd(tree_real, axis=0)
        closest_raw_off_std = np.nanstd(closest_real, axis=0)
        lowest_raw_off_std = np.nanstd(lowest_real, axis=0)

        x_raw_off = np.linspace(0, tree_raw_off.shape[0], tree_raw_off.shape[0]+1)
        bin_width = (x_raw_off[1:] - x_raw_off[:-1]) / 2.
        bin_center = (x_raw_off[:-1] + x_raw_off[1:]) / 2.
        plt.clf()
        plt.errorbar(x=bin_center, y=closest_raw_off, fmt='.', yerr=closest_raw_off_std, xerr=bin_width, label="Closest Binning")
        plt.errorbar(x=bin_center, y=lowest_raw_off, fmt='.', yerr=lowest_raw_off_std, xerr=bin_width, label="Lowest Binning")
        plt.errorbar(x=bin_center, y=tree_raw_off, fmt='.', yerr=tree_raw_off_std, xerr=bin_width, label="Tree Binning")
        plt.legend(loc='best')
        plt.title("PDF - Unfolded / Error For " + str(tree_raw_off.shape[0]) + " Runs")
        plt.xlabel("Bin Number")
        plt.ylabel("log((True - Unf) / Error)")
        plt.yscale('log')
        plt.savefig("output/Multiple_Run_mean.png")
        plt.clf()


        tree_error_pdf = [[],[]]
        print(tree_error_real_pdf)
        print("\n")
        print(tree_error_real_pdf[0])

        tree_error_pdf[0] = tree_real_pdf - tree_error_real_pdf[0]
        tree_error_pdf[1] = tree_error_real_pdf[1] - tree_real_pdf

        tree_raw_off = np.nanmean(tree_real_pdf, axis=0)
        closest_raw_off = np.nanmean(closest_real_pdf, axis=0)
        lowest_raw_off = np.nanmean(lowest_real, axis=0)

        tree_raw_off_std = np.nanstd(tree_real, axis=0)
        closest_raw_off_std = np.nanstd(closest_real, axis=0)
        lowest_raw_off_std = np.nanstd(lowest_real, axis=0)

        x_raw_off = np.linspace(0, tree_raw_off.shape[0], tree_raw_off.shape[0])
        bin_width = (x_raw_off[1:] - x_raw_off[:-1]) / 2.
        bin_center = (x_raw_off[:-1] + x_raw_off[1:]) / 2.
        plt.clf()
        plt.errorbar(x=bin_center, y=closest_raw_off, fmt='.', yerr=closest_raw_off_std, xerr=bin_width, label="Closest Binning")
        plt.errorbar(x=bin_center, y=lowest_raw_off, fmt='.', yerr=lowest_raw_off_std, xerr=bin_width, label="Lowest Binning")
        plt.errorbar(x=bin_center, y=tree_raw_off, fmt='.', yerr=tree_raw_off_std, xerr=bin_width, label="Tree Binning")
        plt.legend(loc='best')
        plt.title("PDF - Unfolded / Error For " + str(tree_raw_off.shape[0]) + " Runs")
        plt.xlabel("Bin Number")
        plt.ylabel("log((True - Unf) / Error)")
        plt.yscale('log')
        plt.savefig("output/Multiple_Run_mean.png")
        plt.clf()

        '''
        Mean over the runs so bin mean, 9 bins on bottom, and mean of those is each one
        true - unfolding / error for true PDF
        unfolding / True for mean of the bins
        Redo bins to have ends, hist with bin center and weights as the values
        The one of true_pdf - unflding -> if have 0 and width of 1, perfect and not biased. If width less than one, overcovered error
        
        '''

        list_svd_tree = np.asarray(list_of_svd_errors[0])
        list_svd_low = np.asarray(list_of_svd_errors[2])
        list_svd_close = np.asarray(list_of_svd_errors[4])

        list_min_tree = np.asarray(list_of_min_errors[0])
        list_min_low = np.asarray(list_of_min_errors[2])
        list_min_close = np.asarray(list_of_min_errors[4])

        output.write("Tree Binning Overall Difference Mean and Std.:\n" + str(np.nanmean(tree_real)) + "\n" + str(
            np.nanstd(tree_real)) + "\n")
        output.write(
            "Closest Binning Overall Difference Mean and Std.:\n " + str(np.nanmean(closest_real)) + "\n" + str(
                np.nanstd(closest_real)) + "\n")
        output.write("Lowest Binning Overall Difference Mean and Std.:\n" + str(np.nanmean(lowest_real)) + "\n" + str(
            np.nanstd(lowest_real)) + "\n")

        output.write(
            "Tree Binning Overall SVD Difference Mean and Std.:\n" + str(np.nanmean(list_svd_tree)) + "\n" + str(
                np.nanstd(list_svd_tree)) + "\n")
        output.write(
            "Closest Binning Overall SVD Difference Mean and Std.:\n " + str(np.nanmean(list_svd_close)) + "\n" + str(
                np.nanstd(list_svd_close)) + "\n")
        output.write(
            "Lowest Binning Overall SVD Difference Mean and Std.:\n" + str(np.nanmean(list_svd_low)) + "\n" + str(
                np.nanstd(list_svd_low)) + "\n")

        output.write(
            "Tree Binning Overall SVD Difference Mean and Std.:\n" + str(np.nanmean(list_min_tree)) + "\n" + str(
                np.nanstd(list_min_tree)) + "\n")
        output.write(
            "Closest Binning Overall SVD Difference Mean and Std.:\n " + str(np.nanmean(list_min_close)) + "\n" + str(
                np.nanstd(list_min_close)) + "\n")
        output.write(
            "Lowest Binning Overall SVD Difference Mean and Std.:\n" + str(np.nanmean(list_min_low)) + "\n" + str(
                np.nanstd(list_min_low)) + "\n")

        output.write("Tree Binning Difference Mean and Std.:\n" + str(np.nanmean(tree_real, axis=1)) + "\n" + str(
            np.nanstd(tree_real, axis=1)) + "\n")
        output.write(
            "Closest Binning Difference Mean and Std.:\n " + str(np.nanmean(closest_real, axis=1)) + "\n" + str(
                np.nanstd(closest_real, axis=1)) + "\n")
        output.write("Lowest Binning Difference Mean and Std.:\n" + str(np.nanmean(lowest_real, axis=1)) + "\n" + str(
            np.nanstd(lowest_real, axis=1)) + "\n")

        output.write(
            "Tree Binning Difference SVD Mean and Std.:\n" + str(np.nanmean(list_svd_tree, axis=1)) + "\n" + str(
                np.nanstd(list_svd_tree, axis=1)) + "\n")
        output.write(
            "Closest Binning Difference SVD Mean and Std.:\n " + str(np.nanmean(list_svd_close, axis=1)) + "\n" + str(
                np.nanstd(list_svd_close, axis=1)) + "\n")
        output.write(
            "Lowest Binning Difference SVD Mean and Std.:\n" + str(np.nanmean(list_svd_low, axis=1)) + "\n" + str(
                np.nanstd(list_svd_low, axis=1)) + "\n")

        output.write(
            "Tree Binning Difference Min Mean and Std.:\n" + str(np.nanmean(list_min_tree, axis=1)) + "\n" + str(
                np.nanstd(list_min_tree, axis=1)) + "\n")
        output.write(
            "Closest Binning Difference Min Mean and Std.:\n " + str(np.nanmean(list_min_close, axis=1)) + "\n" + str(
                np.nanstd(list_min_close, axis=1)) + "\n")
        output.write(
            "Lowest Binning Difference Min Mean and Std.:\n" + str(np.nanmean(list_min_low, axis=1)) + "\n" + str(
                np.nanstd(list_min_low, axis=1)) + "\n")

        tree_error_real_lower = list_mcmc_tree - tree_error_real
        tree_error_real_upper = tree_error_real - list_mcmc_tree

        output.write(
            "Tree Binning MCMC Error Lower Mean and Std.: " + str(np.nanmean(tree_error_real_lower)) + " " + str(
                np.nanstd(tree_error_real_lower)) + "\n")
        output.write(
            "Tree Binning MCMC Error Upper Mean and Std.: " + str(np.nanmean(tree_error_real_upper)) + " " + str(
                np.nanstd(tree_error_real_upper)) + "\n")

        closest_error_real_lower = list_mcmc_close - closest_error_real[0]
        closest_error_real_upper = closest_error_real[1] - list_mcmc_close

        output.write(
            "Closest Binning MCMC Error Lower Mean and Std.: " + str(np.nanmean(closest_error_real_lower)) + " " + str(
                np.nanstd(closest_error_real_lower)) + "\n")
        output.write(
            "Closest Binning MCMC Error Upper Mean and Std.: " + str(np.nanmean(closest_error_real_upper)) + " " + str(
                np.nanstd(closest_error_real_upper)) + "\n")

        lowest_error_real_lower = list_mcmc_low - lowest_error_real[0]
        lowest_error_real_upper = lowest_error_real[1] - list_mcmc_low

        output.write(
            "Lowest Binning MCMC Error Lower Mean and Std.: " + str(np.nanmean(lowest_error_real_lower)) + " " + str(
                np.nanstd(lowest_error_real_lower)) + "\n")
        output.write(
            "Lowest Binning MCMC Error Upper Mean and Std.: " + str(np.nanmean(lowest_error_real_upper)) + " " + str(
                np.nanstd(lowest_error_real_upper)) + "\n")
