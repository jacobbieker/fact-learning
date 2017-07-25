import numpy as np


class Detector:
    '''
    ToyMC detector consisting of n_chambers of length 1. [meter]

    Parameters
    ----------
    n_chambers : int
        Number of chambers

    threshold_chamber : float
        Amount of emitted energy to trigger the chamber

    resolution_chamber : float
        Parameter setting the resolution of the chamber. Interpretation of the
        parameter depends of the implementation and how the particles lose
        their energy.

    energy_loss : ['const', 'random']
        The way the particle losses energy.
            'const' --> constant energy loss see 'loss_rate'
            'random' --> The particle emits part of its energy with a certain
                         probability per meter

    loss_rate : float
        Mean energy loss per meter for 'const'.
        Mean distance between energy losses for 'random'. Every loss is 10% of the total energy
        Maybe it make more sense to let the particle lose a fixed fraction.

    noise : float
        We can also add some kind of noise hits and this parameter
        is the expected noise signal per chamber per event. It is the mean of the normal distribution

    distribution: ['binomial','gaussian']
        The distribution used, with the resolution, to generate the pseudo-observables

    smearing: bool
        Whether to smear the chamber signal or not

    make_noise: bool
        Whether to add random noise or have noise disappear

    plot: bool
        Whether to output various plots of the detector and energy

    Attributes
    ----------
        ???
    '''

    def __init__(self,
                 n_chambers=100,
                 threshold_chambers=1.,
                 resolution_chamber=1.,
                 energy_loss='const',
                 loss_rate=10.,
                 noise=0.,
                 distribution="binomial",
                 smearing=True,
                 make_noise=True,
                 random_state=None):
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state
        self.n_chambers = n_chambers
        self.threshold_chambers = threshold_chambers
        self.resolution_chamber = resolution_chamber
        self.energy_loss = energy_loss
        self.loss_rate = loss_rate
        self.noise = noise
        self.smearing = smearing
        self.distribution = distribution
        self.make_noise = make_noise

    def generate_true_energy_losses(self, energies):
        '''This function should generate the true energy losses in each chamber
        for every event.


        Parameters
        ----------
        energies : array_like, shape=(n_events)
            Starting energy of the particles


        Returns
        -------
        true_hits : array_like, shape=(n_events, n_chambers)
            Array containing the amount of lost energy in each chamber
        '''

        # Need to make sure the mean values are positive from distribution, or just use the means directly

        true_hits = np.zeros(shape=(energies.shape[0], self.n_chambers))

        # For use in the gamma distribution that is strictly positive
        variance = 1.0
        theta = variance / self.loss_rate
        k = self.loss_rate / theta

        if self.energy_loss == 'const':

            for particle_number, energy in enumerate(energies):
                # Create a gamma distribution with a mean of the loss_rate for every chamber
                real_energy_loss = self.random_state.gamma(
                    shape=k, scale=theta, size=self.n_chambers)

                # Energy lost is energy * fraction_lost
                for chamber_number in range(self.n_chambers):
                    # Iterate through the number of chambers and get the energy loss for the chamber
                    energy = energy - real_energy_loss[chamber_number]
                    if energy >= 0.0:
                        # Might need to change the condition, if it exactly loses all its energy, that should show up
                        true_hits[particle_number][chamber_number] = real_energy_loss[chamber_number]
                    else:
                        break

            return true_hits
        elif self.energy_loss == 'random':
            # loss_rate is mean distance travelled between emitting energy
            for particle_number, energy in enumerate(energies):
                # Create a gamma distribution with a mean of the loss_rate for every chamber
                # Create it in the loop to vary it for each particle
                real_distance_travelled = self.random_state.gamma(
                    shape=k, scale=theta, size=self.n_chambers)

                # How much energy the particle loses each time is fixed here as a fraction
                particle_random_loss = 0.1

                # Go through the distance travelled and see where the particle emits energy
                total_distance = 0.0
                for distance in np.nditer(real_distance_travelled):
                    total_distance += distance
                    lost_energy = (energy * particle_random_loss)
                    energy = energy - lost_energy

                    # Get the chamber it happened in
                    chamber_number = int(np.floor(total_distance))

                    if energy >= 0.0 and chamber_number <= self.n_chambers - 1:
                        true_hits[particle_number][chamber_number] = lost_energy

            return true_hits
        else:
            raise ValueError

    def generate_noise(self, n_events):
        '''This function generate noise for each event in each chamber.

        Parameters
        ----------
        n_events : ints
            Number of events


        Returns
        -------
        noise_hits : array_like, shape=(n_events, n_chambers)
            Array containing the amount of lost energy in each chamber
        '''

        noise_hits = np.zeros(shape=(n_events, self.n_chambers))
        if self.make_noise:
            for event_number, event in enumerate(noise_hits):
                noise_distribution = self.random_state.normal(
                    loc=self.noise, size=self.n_chambers)
                zero_threshold = noise_distribution < 0.0
                noise_distribution[zero_threshold] = 0.0
                noise_hits[event_number] = noise_distribution

        return noise_hits

    def generate_chamber_signal(self, chamber_hits):
        '''This function generate the signal of each chamber. It applies some
        kind of smearing and applies the threshold.

        Parameters
        ----------
        chamber_hits : array_like, shape=(n_events, n_chambers)
            Generate hits in for each event in each chamber

        Returns
        -------
        signal : array_like, shape=(n_events, n_chambers)
            Signal return by each chamber in photons
        '''

        signal = np.zeros_like(chamber_hits, dtype=np.float32)

        for particle_number, chambers in enumerate(chamber_hits):
            for chamber_number, energy_value in enumerate(chambers):
                # Energy per photon
                energy_per_photon = 1.0
                # Total photons detected
                total_photons = np.floor(energy_value / energy_per_photon)

                if self.distribution == 'binomial':
                    photons_detected = np.random.binomial(n=total_photons, p=self.resolution_chamber)
                elif self.distribution == 'gaussian':
                    photons_detected = self.random_state.normal(
                        loc=energy_value, scale=self.resolution_chamber)

                if photons_detected < 0.0:
                    photons_detected = 0.0
                # Generate the smearing, based on the energy of the particle received, with std 1/root(N) photons
                if photons_detected > 1.0:
                    if self.smearing:
                        smear = self.random_state.normal(
                            scale=1.0 / np.sqrt(photons_detected))
                        smeared_value = photons_detected + smear
                    else:
                        smeared_value = photons_detected
                else:
                    smeared_value = 0.0

                if smeared_value >= self.threshold_chambers:
                    signal[particle_number][chamber_number] = smeared_value

        return signal

    def simulate(self, energies):
        '''This function returns the signal of each chamber for each events.

        Parameters
        ----------
        energies : array_like, shape=(n_events)
            Starting energy of the particles

        Returns
        -------
        signal : array_like, shape=(n_events, n_chambers)
            Signal return by each chamber
        '''
        true_hits = self.generate_true_energy_losses(energies)
        noise_hits = self.generate_noise(true_hits.shape[0])
        chamber_hits = true_hits + noise_hits
        signal = self.generate_chamber_signal(chamber_hits)
        detector_matrix = self.get_response_matrix(energies, signal)
        return signal, true_hits, energies, detector_matrix

    def get_response_matrix(self, original_energy_distribution, signal, num_bins=50):
        if self.n_chambers < 2:
            raise ValueError("Number of Chambers must be larger than 2")

        # A = np.zeros((self.n_chambers, self.n_chambers))

        # If Acceptance is less than 1, than the normalization should not add up to 1 in that row
        # It should normalize to whatever value the acceptance is, so if its 0.8, should normalize to 0.8


        # To get the response matrix, do a 2d histogram of Original energy distribution vs measured energy, the historgram
        # must be normalized at some point, either row or column, or both, or some other way, but must be normalized
        # Normalized = percentage = probabllilty
        # Make a table, per particle: ID | Observable | True Energy and make 2D histogram of Observable vs True Energy
        #print(original_energy_distribution)
        sum_true_energy_per_particle = original_energy_distribution
        sum_signal = np.sum(signal, axis=1)

        #A, xedge, yedge = np.histogram2d(sum_signal_per_chamber, sum_chamber_per_chamber, normed=False,
        #                                 bins=self.n_chambers)
        #A, xedge, yedge = np.histogram2d(sum_signal_per_chamber, sum_chamber_per_chamber, normed=False)
        A, xedge, yedge = np.histogram2d(sum_true_energy_per_particle, sum_signal, normed=False, bins=num_bins)

        # Normalize based off of the row

        A_row_Norm = A / A.sum(axis=1, keepdims=True)
        # pprint.pprint(A_row_Norm)

        A_col_Norm = A / A.sum(axis=0, keepdims=True)
        # pprint.pprint(A_column_Norm)

        return A_col_Norm
