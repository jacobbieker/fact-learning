fact-learning
-------------
[![Build Status](https://travis-ci.org/jacobbieker/fact-learning.svg?branch=master)](https://travis-ci.org/jacobbieker/fact-learning)


Tasks
-------------

1. Toy MC
		Implement a way to generate a toy simulation. Features we might need:

		1. Takes a Funktion like: N*x^(-gamma) as input

		2. Acceptance: The 'detector' should have a finite probability to miss events. This probability should be dependent on 'x'. Make sure you can activate / deactivate (f(x)=1) the acceptance funtion.
         
		3. At least 1 pseudo-observable which is correlated to 'x'. E.g. something like an energy proxy if 'x' is the true energy.

		- One possible design: Let us assume 'x' is the particle energy and our detector is a series of N chambers. 
		The particles fly through all these chambers with a constant energy loss until the particle lost all its energy 
		(more complicated way is to replace the constant energy lost with randomly occuring energy losses). 
		Each chambers has a detection threshold and measures the energy lost by the particle while it is in the chambers. 
		The total energy lost in the chambers has to be above the threshold and the detected amount of energy is a smeared 
		value of the actual lost energy. With the threshold and the smearing there is a probability > 0 to not detect the particle 
		(required acceptance) the number of triggered chambers and 'signal' detected by those chambers can be used to 
		calculate additional pseudo-observables (e.g. mean signal of all triggered chambers).

		- Keep in mind we need parameters to change the behaviour of our toy MC. 
		In particular we want to make harder/easier to tell which is the true x based on the observables. 
		For the more complex approach those parameters might be the resolution of the chambers and the threshold of the chambers. 
		Perfect would be also have a setting for the detector to have the perfect detector 
		e.g. no smearing no threshold0 --> The detector is able to measure a value that is the true value or rather can be unambiguously calculated from the measured value


2. Unfolding
		1. Unfold with simply inverting the detector response matrix A

				- Plot matrix condition vs. parameters of your toy mc

				- Make plots like https://arxiv.org/pdf/hep-ex/0208022.pdf Fig. 3 for changing number of events in the unfolding and different toy mc settings

				- Unfold and calc errors for multiple drawn datasets (~100 to 500). Calculate/plot the mean/std of (unfolding_result - true_distribution) / unfolding_error . 
				Think about the interpretation of those numbers and what we want to have for a good unfolding and error calculation

				- Investigate the impact of regularization via eigenvalue cut-off

		2. Repeat 1. now with SVD unfolding
				- In cases you have more than 1 observable try to use more than 1 in the unfolding (2d binning)

				- Investigate the impact of regularization via singular cut-off (maybe also the effect of a dampening of small eigenvalues**)

		3. Unfolding via LLH (Log-LiklieHood) fit:
				- Implement the LLH, gradient, Hesse-Matrix without regularization and try to find the minima with a self implemented gradient descent

				- Use the implemented LLH (gradient, Hesse-Matrix) and try to solve it with scipy.optimize.minimize

				- Calculate the errors using the Hesse Matrix - check

				- Try to do a scan (build the LLH landscape) and use Wilk"s theorem to do obtain the errors --> unfold only ~4 Bins to get the dimensionality low

				- Use a MCMC (e.g. emcee) for the minimization**

3. Forwardfolding

	1. To be continued

