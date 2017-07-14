import numpy as np
from numpy.random import normal, gamma
import matplotlib.pyplot as plt


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
        Mean distance between energy losses for 'random'.
        Maybe it make more sense to let the particle lose a fixed fraction.

    noise : float
        We can also add some kind of noise hits and this parameter
        is the expected noise signal per chamber per event.

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
                 smearing=True,
                 plot=False):

        self.n_chambers = n_chambers
        self.threshold_chambers = threshold_chambers
        self.resolution_chamber = resolution_chamber
        self.energy_loss = energy_loss
        self.loss_rate = loss_rate
        self.noise = noise
        self.smearing = smearing
        self.plot = plot

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

        true_hits = np.zeros(shape=(energies.shape, self.n_chambers))

        # For use in the gamma distribution that is strictly positive
        variance = 1.0
        theta = variance / self.loss_rate
        k = self.loss_rate / theta

        if self.energy_loss == 'const':

            for particle_number, energy in enumerate(energies):
                # Create a gamma distribution with a mean of the loss_rate for every chamber
                real_energy_loss = gamma(shape=k, scale=theta, size=self.n_chambers)

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
                real_distance_travelled = gamma(shape=k, scale=theta, size=self.n_chambers)

                # How much energy the particle loses each time is fixed here as a fraction
                particle_random_loss = 10.0 / 100.0

                # Go through the distance travelled and see where the particle emits energy
                total_distance = 0.0
                for distance in np.nditer(real_distance_travelled):
                    total_distance += distance
                    lost_energy = (energy * particle_random_loss)
                    energy = energy - lost_energy

                    # Get the chamber it happened in
                    chamber_number = np.floor(total_distance)

                    if energy >= 0.0:
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

        # Idea: The self.noise parameter sets the mean + variance? or max? in each detector chamber
        # Supposed to be expected signal, so can be around there.
        # Currently, detectors have a resolution of 1.0, the entire detector is essentially 1 pixel
        # So max should just shift it up by the expected amount?
        # And if there are more than 1 pixel, it should be randomly distributed between them all
        # So, based off the self.resolution_chamber, noise gets distributed into that many bins
        # but that doesn't fit in with the noise_hits shape, since the shape is not three dimensional

        noise_hits = np.zeros(shape=(n_events.shape, self.n_chambers))

        # If we go with a higher resolution means that the noise is randomly distributed around the detector...

        for event in n_events:
            for chamber in range(self.n_chambers):
                noise_level = self.noise
                # If using resolution as different parts, one more for loop under this line
                location_noise = np.random.uniform(high=noise_level)
                noise_hits[event][chamber] = location_noise
                # If using resolution to split chambers, here is where the subtract leaves the rest of the noise level
                # to be distributed to the other chambers, last one getting the remainder
                noise_level -= location_noise

        raise NotImplementedError

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
            Signal return by each chamber
        '''

        signal = np.zeros_like(chamber_hits)

        for particle_number, chambers in enumerate(chamber_hits):
            for chamber_number, energy_value in enumerate(chambers):
                # Generate the smearing, based on the energy of the particle received, with std 1/root(N) energy
                if self.smearing:
                    smear = normal(scale=1.0 / np.sqrt(energy_value))
                    smeared_value = energy_value + smear[chamber_number]
                else:
                    smeared_value = energy_value

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
        psuedo_observables = self.generate_pseudo_observables(signal)
        if self.plot:
            self.plot_simulation(energies, true_hits, noise_hits, chamber_hits, signal, psuedo_observables)
        return signal

    def plot_simulation(self, energies, true_hits, noise_hits, chamber_hits, signal, psuedo_observables):
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
        plt.plot(true_hits[1], signal[1])
        plt.xlabel("True hits")
        plt.ylabel("Signal")
        plt.show()

        plt.plot(signal[1] / true_hits[1])
        plt.title("Signal / True Hits")
        plt.show()

        # Plotting the psuedo-observables vs the true energy hits
        plt.plot(true_hits[1], psuedo_observables[1])
        plt.xlabel("True Energy")
        plt.ylabel("Psuedo-Observables")
        plt.show()



        # Plot
        raise NotImplementedError

    def generate_pseudo_observables(self, signal):
        '''This function produces pseudo-observables from the signal generated by each chamber

        Currently only create a single psuedo observable based directly on the signal

        Parameters
        ----------
        signal: array_like, shape=(n_events, n_chambers)
            Signal returned by each chamber


        Returns
        -------
        pseudo_observables: array_like, shape=(n_observable, n_chambers)
            Psuedo-observables returned by each chamber
        '''

        energy_for_hit = 1.0  # Energy required for each "hit" on the detector
        # Psuedo-observables are only one observable at the moment
        psuedo_observables = np.zeros_like(signal)

        for particle_number, chambers in enumerate(signal):
            for chamber_number, energy_value in enumerate(chambers):
                number_of_hits = np.floor(energy_value / energy_for_hit)
                psuedo_observables[particle_number][chamber_number] = number_of_hits

        return psuedo_observables
