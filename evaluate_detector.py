import numpy as np

import matplotlib.pyplot as plt


def plot_response_matrix(A):
    plt.imshow(A, interpolation="nearest", origin="upper")
    plt.xscale('log')
    plt.yscale('log')
    plt.colorbar()
    plt.title("Detector Response Matrix Raw")
    plt.show()


def plot_simulation(detector,
                    energies,
                    true_hits,
                    noise_hits,
                    chamber_hits,
                    signal):
    ''' This function produces various plots from the simulation

        Parameters
        ----------
        energies :
        true_hits:
        noise_hits:
        chamber_hits:
        signal:

        Returns
        -------
        Nothing. Displays multiple plots
        '''
    plt.hist(energies, 40)
    plt.title("Particle Energies")
    plt.xlabel("Energy")
    plt.ylabel("Number of Particles")
    plt.show()

    figure = plt.figure()

    ax = figure.add_subplot(111)

    for index in range(energies.shape[0]):
        ax.scatter(true_hits[index], signal[index])

    # plt.scatter(true_hits[1], signal[1])
    plt.xlabel("True Energy")
    plt.ylabel("Signal (Number of Photons)")
    plt.title("True Energy vs Signal for each Chamber and each Event")
    plt.show()

    # Now plot energy per bin
    figure = plt.figure()
    ax = figure.add_subplot(111)

    for index in range(energies.shape[0]):
        ax.scatter(np.arange(0, detector.n_chambers), true_hits[index])
        # ax.scatter(np.arange(0, detector.n_chambers), signal[index])
    plt.title("True Energy per Chamber per Event")
    plt.ylabel("Energy Value")
    plt.xlabel("Chamber Number")
    plt.show()

    # Now plot energy per bin
    figure = plt.figure()
    ax = figure.add_subplot(111)

    for index in range(energies.shape[0]):
        # ax.scatter(np.arange(0,detector.n_chambers), true_hits[index])
        ax.scatter(np.arange(0, detector.n_chambers), signal[index])
    plt.title("Signal per Chamber per Event")
    plt.ylabel("Number of Photons")
    plt.xlabel("Chamber Number")
    plt.show()

    figure = plt.figure()
    ax = figure.add_subplot(111)

    sum_chambers = np.sum(true_hits, axis=1)
    sum_signal = np.sum(signal, axis=1)

    ax.hist(sum_chambers, bins=20, label="True Energy")
    ax.hist(sum_signal, bins=20, label="Measured Photons")
    plt.legend(loc='best')
    plt.show()

    plt.hist(sum_chambers, bins=np.linspace(min(sum_chambers), max(sum_chambers), 50), normed=True,
             label="True Energy")
    plt.hist(sum_signal, bins=np.linspace(min(sum_signal), max(sum_signal), 50),
             histtype='step', normed=True, label="Measured Photons")
    plt.legend(loc='best')
    plt.title("True Energy Distribution vs Measured Distribution")
    plt.ylabel("Normalized Value")
    plt.xlabel("Total Energy Deposited per Particle")
    plt.show()

    # Detector response matrix: 2d histogram
    sum_chamber_per_chamber = np.sum(true_hits, axis=0)
    sum_signal_per_chamber = np.sum(signal, axis=0)

    plt.hist2d(sum_chamber_per_chamber, sum_signal_per_chamber, normed=True)
    plt.title("Detector Response Matrix for summed per chamber")
    plt.xlabel("True Energy")
    plt.ylabel("Measured Energy")
    plt.show()

    plt.hist2d(sum_chamber_per_chamber, sum_signal_per_chamber, normed=False)
    plt.title("Detector Response Matrix for summed per chamber (Unnormalized)")
    plt.xlabel("True Energy")
    plt.ylabel("Measured Energy")
    plt.show()

    plt.hist2d(sum_chambers, sum_signal, normed=True)
    plt.title("Detector Response Matrix for summed per particle")
    plt.xlabel("True Energy")
    plt.ylabel("Measured Energy")
    plt.show()

    # Number of events at that energy level for both True and Measured

    if detector.make_noise:
        plt.hist(noise_hits, 10)
        plt.title("Noise Distribution Per Chamber")
        plt.show()
