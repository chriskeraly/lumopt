""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import numpy as np

from lumopt.optimizers.generic_optimizers import Optimizer

class AdaptiveGradientDescent(Optimizer):
    """ Almost identical to FixedStepGradientDescent, except that dx changes according to the following rule:

        dx = min(max_dx,dx*dx_regrowth_factor)
        while newfom < oldfom
        dx = dx / 2
        if dx < min_dx:
            dx = min_dx
            return newfom
    """

    def __init__(self, max_dx, min_dx, max_iter, dx_regrowth_factor, all_params_equal, scaling_factor):
        '''
        :param max_dx:             maximum allowed change of a parameter per iteration.
        :param min_dx:             minimum step size (for the largest parameter changing) allowed.
        :param dx_regrowth_factor: by how much dx will be increased at each iteration.
        :param max_iter:           maximum number of iterations to run.
        :param all_params_equal:   if true, all parameters will be changed by +/- dx depending on the sign of their associated shape derivative.
        :param scaling_factor:     scaling factor to bring the optimization variables close to one.
        '''

        super(AdaptiveGradientDescent,self).__init__(max_iter, scaling_factor)

        self.max_dx = max_dx * self.scaling_factor
        self.all_params_equal = all_params_equal
        self.predictedchange_hist = []
        self.min_dx = min_dx * self.scaling_factor
        self.dx_regrowth_factor = dx_regrowth_factor
        self.dx = self.max_dx

    def run(self):
        self.current_params = self.start_point
        self.current_fom = self.callable_fom(self.current_params)
        gradients = self.callable_jac(self.current_params)
        self.callback()
        while self.iteration < self.max_iter:
            self.dx = np.minimum(self.dx * self.dx_regrowth_factor, self.max_dx)
            new_params = self.current_params + self.calculate_change(gradients, self.dx)
            new_params = self.enforce_bounds(new_params)
            new_fom = self.callable_fom(new_params)
            while new_fom < self.fom_hist[-1] and self.dx > self.min_dx:
                self.reduce_step_size()
                new_params = self.current_params + self.calculate_change(gradients, self.dx)
                new_params = self.enforce_bounds(new_params)
                new_fom = self.callable_fom(new_params)
            if self.dx == self.min_dx:
                print('dx at mindx: forcing update')
            self.current_params = new_params
            self.current_fom = new_fom
            if self.iteration != self.max_iter:
                gradients = self.callable_jac(self.current_params)
            self.callback()

    def calculate_change(self, gradients, dx):
        if self.all_params_equal:
            change = ((np.array(gradients) > 0.0) * 2.0 - 1.0)*dx
        else:
            change = np.array(gradients)/np.max(np.abs(np.array(gradients)))*dx
        return change

    def reduce_step_size(self):
        self.dx = np.maximum(self.dx / 2.0, self.min_dx)
        print('Figure of merit decreasing: reducing step size to {}'.format(self.dx))

    def enforce_bounds(self,params):
        bounds_min = np.array([bound[0] for bound in self.bounds])
        bounds_max = np.array([bound[1] for bound in self.bounds])
        return np.maximum(bounds_min, (np.minimum(bounds_max, params)))