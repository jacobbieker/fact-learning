import numpy as np
import matplotlib.pyplot as plt


def plot_true_vs_signal(true_vector, detected_signal, energies, num_bins=20):
    plt.bar(true_vector[1][:-1], true_vector[0], width=true_vector[1][1:], label="Y Vector")
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


def plot_unfolded_vs_true(true_vector, unfolded_vector, energies, num_bins=20):
    # plt.hist(unfolded_vector, bins=num_bins, label="Unfolded Energy")
    plt.bar(true_vector[1][:-1], unfolded_vector, width=true_vector[1][1:], label="Unfolded Energy")
    plt.hist(energies, bins=np.linspace(min(energies), max(energies), num_bins), normed=False,
             label="True Energy", histtype='step')
    y_values = np.histogram(unfolded_vector, bins=len(unfolded_vector))
    # plt.errorbar(unfolded_vector, y=y_values[0], yerr=sigma_x_unf)
    plt.title("Number of Particles: " + str(energies.shape[0]))
    plt.legend(loc='best')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
