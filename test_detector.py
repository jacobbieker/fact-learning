import numpy as np
from detector import Detector


def test_detector(random_state=None, plot=False):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    energies = 1000.0 * random_state.power(0.70, 500)
    # energies = normal(loc=1000.0, scale=500, size=1000)
    below_zero = energies < 0.0
    energies[below_zero] = 1.0

    detector = Detector(distribution='gaussian',
                        energy_loss='const',
                        make_noise=True,
                        smearing=True,
                        resolution_chamber=1.,
                        noise=0.,
                        random_state=random_state)
    signal, true_hits, energies_return, detector_matrix = detector.simulate(
        energies)

    assert signal.shape[0] == energies.shape[0]
    assert signal.shape[1] == detector.n_chambers
    assert true_hits.shape[0] == energies.shape[0]
    assert true_hits.shape[1] == detector.n_chambers
    assert all(energies_return == energies)
    assert all(np.isclose(np.sum(detector_matrix, axis=0), 1.))


def test_consistency(random_state=None):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    energies = 1000.0 * random_state.power(0.70, 5000)
    below_zero = energies < 1.0
    energies[below_zero] = 1.0

    signal_valuess = []
    true_hitss = []
    energies_returns = []
    detector_matrixs = []
    for i in range(100):
        detector = Detector(distribution='gaussian',
                            energy_loss='const',
                            make_noise=True,
                            smearing=True,
                            resolution_chamber=1.,
                            noise=0.,
                            response_bins=20,
                            rectangular_bins=20,
                            random_state=random_state)
        signal, true_hits, energies_return, detector_matrix = detector.simulate(
            energies)
        signal_valuess.append(signal)
        true_hitss.append(true_hits)
        energies_returns.append(energies_return)
        detector_matrixs.append(detector_matrix)

    signal_valuess = np.asarray(signal_valuess)
    true_hitss = np.asarray(true_hitss)
    energies_returns = np.asarray(energies_returns)
    detector_matrixs = np.asarray(detector_matrixs)

    for index, array in enumerate(signal_valuess):
        print(index)
        print(signal_valuess)
        print(signal_valuess[index+1])
        assert np.array_equal(signal_valuess[index], signal_valuess[index+1])
        assert np.array_equal(true_hitss[index], true_hitss[index+1])
        assert np.array_equal(energies_returns[index], energies_returns[index+1])
        assert np.array_equal(detector_matrixs[index], detector_matrixs[index+1])

if __name__ == "__main__":
    test_consistency(1347)
    test_detector(1347, plot=True)
