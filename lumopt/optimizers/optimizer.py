""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import copy
import numpy as np
import scipy as sp

class Optimizer(object):
    """ Base class (or super class) for all optimizers. """

    def __init__(self, max_iter, scaling_factor, target_fom = 0, scale_initial_gradient_to = None):
        """ Most optimizers assume the variables to optimize are roughly of order of magnitude of one. Since geometry 
            parameters are usually of the order 1e-9 to 1e-6, it can be useful to scale them to have a magnitude close to one.
            
            :param max_iter:       maximum number of iterations.
            :param scaling_factor: scalar or vector of the same length as the optimization parameters; typically used to 
                                   scale the optimization parameters so that they have magnitudes in the range zero to one.
            :param target_fom:     A target value for the figure of merit. This allows to print/plot the distance of the current
                                   design from a target value
        """

        self.max_iter = max_iter
        self.scaling_factor = np.array(scaling_factor).flatten()
        self.fom_scaling_factor = 1
        self.target_fom = target_fom
        self.current_fom = None
        self.current_gradients = []
        self.current_params = []
        self.fom_hist = []
        self.params_hist = []
        self.gradients_hist = []
        self.iteration = 0
        self.scale_initial_gradient_to = scale_initial_gradient_to
        self.fom_scaling_factor=1

    def initialize(self, start_params, callable_fom, callable_jac, bounds, plotting_function):
        """ Loads the scaled starting point, the bounds and the callables to be used in the optimizer."""

#        assert bounds.shape[0] == start_params.size and bounds.shape[1] == 2
        assert self.scaling_factor.size == 1 or self.scaling_factor.size == start_params.size
        self.callable_fom, self.callable_jac = self.define_callables(callable_fom, callable_jac)
        self.bounds = bounds * self.scaling_factor.reshape((self.scaling_factor.size, 1))
        self.define_callback(plotting_function)
        
        self.reset_start_params(start_params, self.scale_initial_gradient_to)

    def reset_start_params(self, start_params, min_required_rel_change):
        assert self.bounds.shape[0] == start_params.size
        self.start_point = start_params * self.scaling_factor
        if min_required_rel_change is not None:
            self.auto_detect_scaling(min_required_rel_change)
        else:
            self.fom_scaling_factor = 1

    def define_callables(self, callable_fom, callable_jac):
        """ This makes the function that is called by optimzation methods."""

        def callable_fom_local(params):
            fom = callable_fom(params / self.scaling_factor)
            return fom * self.fom_scaling_factor

        def callable_jac_local(params):
            gradients = callable_jac(params / self.scaling_factor) / self.scaling_factor
            self.current_gradients = gradients
            return gradients * self.fom_scaling_factor

        return callable_fom_local, callable_jac_local

    def auto_detect_scaling(self, min_required_rel_change):
            # Calculate the actual epsilon change 
            params = self.start_point
            gradients = self.callable_jac(params)
            params2 = (params - gradients)
            bounds_min = np.array([bound[0] for bound in self.bounds])
            bounds_max = np.array([bound[1] for bound in self.bounds])
            clamped_params = np.maximum(bounds_min, (np.minimum(bounds_max, params2)))
            actual_params = params - clamped_params
            max_change = max(abs(actual_params))
            self.fom_scaling_factor = min_required_rel_change / max_change
            print("Scaling factor is {}".format(self.fom_scaling_factor))

    def define_callback(self, plotting_function):
        def callback(*args):
            """ Called at the end of each iteration to record results."""
            self.params_hist.append(copy.copy(self.current_params))
            self.fom_hist.append(self.current_fom) 
            self.gradients_hist.append(copy.copy(self.current_gradients))
            self.iteration += 1
            plotting_function()
            self.report_writing()
        self.callback = callback

    def plot(self, fomax, paramsax, gradients_ax):
        fomax.clear()

        if self.target_fom == 0:
            fomax.plot(range(self.iteration),np.abs(self.fom_hist))#< For visualization purposes we take the absolute value even if the actual FOM is negative
        else:
            fomax.semilogy(range(self.iteration),np.abs(self.fom_hist))
        
        fomax.set_xlabel('Iteration')
        fomax.set_title('Figure of Merit')
        fomax.set_ylabel('FOM')

        if paramsax is not None:
            paramsax.clear()
            paramsax.semilogy(range(self.iteration),np.abs(self.params_hist))
            paramsax.set_xlabel('Iteration')
            paramsax.set_ylabel('Parameters')
            paramsax.set_title("Parameter evolution")
    
        if (gradients_ax is not None) and hasattr(self, 'gradients_hist'):
            gradients_ax.clear()
            gradients_ax.semilogy(range(self.iteration),np.abs(self.gradients_hist))
            gradients_ax.set_xlabel('Iteration')
            gradients_ax.set_ylabel('Gradient Magnitude')
            gradients_ax.set_title("Gradient evolution")

    def report_writing(self):
        with open('optimization_report.txt','a') as f:
            f.write('AT ITERATION {}:  FOM = {}\n'.format(self.iteration,self.fom_hist[-1]))
            f.write('PARAMETERS = {}\n'.format(self.params_hist[-1]/self.scaling_factor))
            f.write('\n \n')