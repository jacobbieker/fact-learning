class Detector:
    '''
    ToyMC detector consisting of n_chambers of length 1.


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
            'random' --> The particle emitts part of its energy with a certain
                         propability per meter

    loss_rate : float
        Mean energy loss per meter for 'const'.
        Mean distance between energy losses for 'random'.
        Maybe it make more sense to let the particle loss a fixed fraction.

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
                 noise=0.):
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def generate_chamber_signal(chamber_hits):
        '''This function generate the singal of each chamber. It applies some
        kind of smearing and applies the threshold.

        Parameters
        ----------
        true_hits : array_like, shape=(n_events, n_chambers)
            Generate hits in for each event in each chamber

        Returns
        -------
        signal : array_like, shape=(n_events, n_chambers)
            Signal return by each chamber
        '''
        raise NotImplementedError

    def simulate(self, energies):
        '''This function returns the singal of each chamber for each events.

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
        return self.generate_chamber_signal(chamber_hits)
