""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import numpy as np
import numpy.random

from lumopt.optimizers.generic_optimizers import Optimizer

class FixedStepGradientDescent(Optimizer):
    """ Gradient descent with the option to add noise, and parameter scaling

        Update Equation:

            \Delta p_i = \frac{\frac{dFOM}{dp_i}}{max_j(|\frac{dFOM}{dp_j}|)}\Delta x +\text{noise}_i

            if all_params_equal = True

            \Delta p_i = sign(\frac{dFOM}{dp_i})\Delta x +\text{noise}_i

        Noise can be added in the update equation, if the optimization has many local optima: noise = rand([-1,1])*noise_magnitude.

    """

    def __init__(self, max_dx, max_iter, all_params_equal, noise_magnitude, scaling_factor):
        """
        :param max_dx:           maximum allowed change of a parameter per iteration.
        :param max_iter:         maximum number of iterations to run.
        :param all_params_equal: if true, all parameters will be changed by +/- dx depending on the sign of their associated shape derivative.
        :param noise_magnitude:  amplitude of the noise.
        :param scaling_factor:   scaling factor to bring the optimization variables closer to 1
        """

        super(FixedStepGradientDescent, self).__init__(max_iter, scaling_factor)

        self.max_dx = max_dx * self.scaling_factor
        self.all_params_equal = all_params_equal
        self.noise_magnitude = noise_magnitude * self.scaling_factor
        self.predictedchange_hist = []

    def run(self):
        self.current_params = self.start_point
        while self.iteration < self.max_iter:
            current_fom = self.callable_fom(self.current_params)
            self.current_fom = current_fom
            gradients = self.callable_jac(self.current_params)
            change = self.calculate_change(gradients, self.max_dx)
            self.current_params += change
            self.add_noise()
            self.current_params = self.enforce_bounds(self.current_params)
            self.predictedchange_hist.append(sum(gradients * change))
            self.callback()

    def calculate_change(self, gradients, dx):
        if self.all_params_equal:
            change = ((np.array(gradients) > 0.0) * 2.0 - 1.0)*dx
        else:
            change = np.array(gradients)/np.max(np.abs(np.array(gradients)))*dx
        return change

    def add_noise(self):
        noise = self.noise_magnitude*(np.random.rand(len(self.current_params)) - 0.5) * 2.0
        self.current_params = self.current_params + noise

    def enforce_bounds(self,params):
        bounds_min = np.array([bound[0] for bound in self.bounds])
        bounds_max = np.array([bound[1] for bound in self.bounds])
        return np.maximum(bounds_min, (np.minimum(bounds_max, params)))