import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import powerlaw
import corner


def plot_true_vs_signal(true_vector, detected_signal, energies, num_bins=20):
    plt.bar(true_vector[1][:-1], true_vector[0], width=true_vector[1][1:], fill=False, label="Y Vector")
    plt.hist(energies, bins=np.linspace(min(energies), max(energies), num_bins), normed=False,
             label="True Energy", histtype='step')
    plt.hist(detected_signal,
             bins=np.linspace(min(detected_signal), max(detected_signal), num_bins), normed=False,
             label="Summed Signal Energy", histtype='step')
    # Problem is that it is already binned, so this is just binning it again. Still not sure why it gets so large
    # But for it being small, probably because its a bin of bins, so that's why its at 1 or 4 or things like that
    plt.title("True Energy vs Y_Vector")
    plt.legend(loc='best')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def plot_corner(unfolded_vector, energies=None, title=None):
    figure = corner.corner(unfolded_vector, quantiles=[0.16],
                           show_titles=True)
    plt.plot(figure)
    plt.show()

def plot_unfolded_vs_true(unfolded_vector, energies, errors=None, num_bins=20, title=None):
    # plt.hist(unfolded_vector, bins=num_bins, label="Unfolded Energy")
    # sum_true_energy = np.sum(energies, axis=1)
    binning = np.linspace(min(energies), max(energies), num_bins + 1)
    bin_width = (binning[1:] - binning[:-1]) / 2.
    bin_center = (binning[:-1] + binning[1:]) / 2.
    true_hits = np.histogram(energies, bins=binning)
    if errors is not None:
        plt.errorbar(bin_center, unfolded_vector, xerr=bin_width, yerr=errors, fmt='.', label="Unfolding Error")

    plt.hist(bin_center, weights=true_hits[0], bins=binning, normed=False,
             label="True Energy", histtype='step')
    plt.hist(bin_center, bins=binning, weights=unfolded_vector, histtype='step', label='Unfolded Energy')
    x_pdf_space = np.linspace(powerlaw.ppf(0.01, 0.70), powerlaw.ppf(1.0, 0.70), unfolded_vector.shape[0])
    x_vector = powerlaw.pdf(x_pdf_space, 0.70)
    plt.plot(1000.0 * x_pdf_space, x_vector, 'r-', lw=5, alpha=0.6, label='powerlaw pdf')
    # plt.errorbar(unfolded_vector, y=y_values[0], yerr=sigma_x_unf)
    if not title:
        plt.title("Number of Particles: " + str(energies.shape[0]))
    else:
        plt.title(str(title))
    plt.legend(loc='best')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


def plot_unfolded_vs_signal_vs_true(unfolded_vector, signal, energies, errors=None, num_bins=20, title=None):
    # plt.hist(unfolded_vector, bins=num_bins, label="Unfolded Energy")
    # sum_true_energy = np.sum(energies, axis=1)
    binning = np.linspace(min(energies), max(energies), num_bins + 1)
    bin_width = (binning[1:] - binning[:-1]) / 2.
    bin_center = (binning[:-1] + binning[1:]) / 2.
    true_hits = np.histogram(energies, bins=binning)
    if errors is not None:
        plt.errorbar(bin_center, unfolded_vector, xerr=bin_width, yerr=errors, fmt='.', label="Unfolding Error")

    plt.hist(bin_center, weights=energies, bins=binning, normed=False,
             label="True Energy", histtype='step')
    plt.hist(bin_center, bins=binning, weights=unfolded_vector, histtype='step', label='Unfolded Energy')
    plt.hist(bin_center, bins=binning, weights=signal, histtype='step', label='Signal')
    # x_pdf_space = np.linspace(powerlaw.ppf(0.01, 0.70), powerlaw.ppf(1.0, 0.70), unfolded_vector.shape[0])
    # x_vector = powerlaw.pdf(x_pdf_space, 0.70)
    # plt.plot(1000.0 * x_pdf_space, x_vector, 'r-', lw=5, alpha=0.6, label='powerlaw pdf')
    # plt.errorbar(unfolded_vector, y=y_values[0], yerr=sigma_x_unf)
    if not title:
        plt.title("Number of Particles: " + str(np.sum(energies)))
    else:
        plt.title(str(title))
    plt.legend(loc='best')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


def plot_eigenvalues(eigenvalues, eigenvectors, n_dims):
    eigen_vals = np.absolute(eigenvalues)
    sorting = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[sorting]
    kappa = max(eigen_vals) / min(eigen_vals)
    inv_eigen_vals = 1 / eigen_vals
    highest_amp = np.max(np.absolute(inv_eigen_vals))

    plt.bar(range(n_dims), inv_eigen_vals,
            label=u'$\kappa$ = %.2f $\lambda_{%d}$ = %.2f' % (kappa, n_dims, highest_amp))
    plt.xlabel('Index $j$')
    plt.ylabel('Value of inverse Eigenvalue')
    plt.title('Ordered Inverse Eigenvalues')
    plt.legend(loc='best')
    plt.show()

    U = eigenvectors[:, sorting]
    f, axes = plt.subplots(U.shape[1], sharex=True, sharey=True, figsize=(50, 4 * U.shape[1]))
    for i, ax_i in enumerate(axes):
        ax_i.bar(range(n_dims), U[:, i], color='C1', label='Eigenvector %d' % i)
        # ax_i.set_axis_off()
        ax_i.set_title('Shape Eigenvectors {}'.format(i))
        ax_i.set_xlabel('Index $j$')
    plt.show()


def plot_eigenvalue_coefficients(true_coefficients, folded_coefiicient, measured_coefficients, error):
    true_b = true_coefficients
    folded_b_j = folded_coefiicient
    measured_c = measured_coefficients

    folded_b = np.sort(np.abs(folded_b_j))
    true_b = np.sort(np.abs(true_b))
    measured_c = np.sort(np.abs(measured_c))

    folded_b[:] = folded_b[::-1]
    true_b[:] = true_b[::-1]
    measured_c[:] = measured_c[::-1]

    binning = np.linspace(0, true_b.shape[0], true_b.shape[0] + 1)
    bin_center = (binning[:-1] + binning[1:]) / 2

    plt.hist(bin_center, weights=true_b, bins=binning, histtype='step', label="True B")
    plt.hist(bin_center, weights=folded_b, bins=binning, histtype='step', label="Folded B_j")
    plt.axhline(y=np.mean(error), xmin=0, xmax=1.0)

    plt.xlabel("Index j")
    plt.title("Coeff of true and folded distributions")
    plt.legend(loc="best")
    plt.yscale('log')
    # plt.xscale('log')
    plt.show()

    plt.hist(bin_center, weights=measured_c, bins=binning, histtype='step', label="Measured C")

    plt.xlabel("Index j")
    plt.title("Coeff of measured distribution")
    plt.legend(loc="best")
    plt.yscale('log')
    # plt.xscale('log')
    plt.show()


def plot_error_stats(mean, std):
    x = np.linspace(0, mean.shape[0], num=mean.shape[0])
    plt.errorbar(x, mean, yerr=std, linestyle='None')
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, mean, 1))(np.unique(x)))
    plt.title("Mean and Std of errors for different runs")
    plt.ylabel("Mean/Standard Deviation")
    plt.xlabel("Run Number")
    plt.show()


def plot_svd_parts(d, s, z):
    d = np.abs(d)
    binning = np.linspace(0, d.shape[0], d.shape[0] + 1)
    bin_center = (binning[:-1] + binning[1:]) / 2

    plt.hist(bin_center, weights=d, bins=binning, histtype='step', label="|D|")
    plt.legend(loc='best')
    plt.yscale('log')
    plt.show()
    plt.hist(bin_center, weights=s, bins=binning, histtype='step', label="S")
    plt.legend(loc='best')
    plt.yscale('log')
    plt.show()
