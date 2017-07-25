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


if __name__ == "__main__":
    test_detector(1347, plot=True)
