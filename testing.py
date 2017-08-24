import numpy as np

from matplotlib import pyplot as plt
from funfolding import binning, model
from scipy import linalg


def SVD_Unf(model, vec_y, vec_x):
    A = model.A
    m = A.shape[0]
    n = A.shape[1]
    U, S_values, V_T = linalg.svd(A)
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
    X_validate = df_validate.get(unfolding_cols).values
    y_validate = df_validate.get('gamma.energy').values
    X_train = df_train.get(unfolding_cols).values
    y_train = df_train.get('gamma.energy').values

    binning_E = np.linspace(2.1, 6.0, 14)

    binned_E_validate = np.digitize(y_train, binning_E)
    binned_E_train = np.digitize(y_validate, binning_E)

    tree_binning = binning.TreeBinningSklearn(
        regression=False,
        max_features=None,
        min_samples_split=2,
        max_depth=None,
        min_samples_leaf=100,
        max_leaf_nodes=None,
        random_state=1337)

    tree_binning.fit(
        X_train,
        binned_E_train)

    binned_g_validate = tree_binning.digitize(X_validate)

    tree_binning_model = model.LinearModel()
    tree_binning_model.initialize(
        digitized_obs=binned_g_validate,
        digitized_truth=binned_E_validate)

    vec_y, vec_x = tree_binning_model.generate_vectors(
        digitized_obs=binned_g_validate,
        digitized_truth=binned_E_validate)

    vec_x_est, V_x_est, vec_b, sigma_b, vec_b_est, s_values = SVD_Unf(
        tree_binning_model, vec_y, vec_x)

    normed_b = np.absolute(vec_b / sigma_b)
    normed_b_est = np.absolute(vec_b_est / sigma_b)
    order = np.argsort(normed_b)[::-1]

    normed_b = normed_b[order]
    normed_b_est = normed_b_est[order]
    fig, ax = plt.subplots()
    binning = np.linspace(0, len(normed_b), len(normed_b) + 1)
    bin_centers = (binning[1:] + binning[:-1]) / 2
    bin_width = (binning[1:] - binning[:-1]) / 2

    ax.hist(bin_centers,
            bins=binning,
            weights=normed_b,
            label='Truth',
            histtype='step')
    ax.hist(bin_centers, bins=binning, weights=normed_b_est, label='Unfolded',
            histtype='step')
    ax.axhline(1.)
    ax.set_xlabel(r'Index $j$')
    ax.set_ylabel(r'\left|b_j/\sigma_j\right|')
    ax.set_ylim([1e-2, 1e2])
    ax.set_yscale("log", nonposy='clip')
    ax.legend(loc='best')
    fig.savefig('08_classic_binning.png')