""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import copy
import numpy as np
import scipy as sp

class Optimizer(object):
    """ Base class (or super class) for all optimizers. """

    def __init__(self, max_iter, scaling_factor):
        """ Most optimizers assume the variables to optimize are roughly of order of magnitude of one. Since geometry 
            parameters are usually of the order 1e-9 to 1e-6, it can be useful to scale them to have a magnitude close to one.
            
            :param max_iter:       maximum number of iterations.
            :param scaling_factor: factor for scaling optimization parameters before passing them to the optimizer. 
        """

        self.max_iter = max_iter
        self.scaling_factor = scaling_factor
        self.current_fom = None
        self.current_gradients = []
        self.current_params = []
        self.fom_hist = []
        self.params_hist = []
        self.gradients_hist = []
        self.iteration = 0

    def initialize(self, start_params, callable_fom, callable_jac, bounds, plotting_function):
        """ Loads the scaled starting point, the bounds and the callables to be used in the optimizer."""

        self.start_point = start_params * self.scaling_factor
        self.callable_fom, self.callable_jac = self.define_callables(callable_fom,callable_jac)
        self.bounds = bounds * self.scaling_factor
        self.define_callback(plotting_function)

    def define_callables(self, callable_fom, callable_jac):
        """ This makes the function that is called by optimzation methods."""

        def callable_fom_local(params):
            fom = callable_fom(params / self.scaling_factor)
            return fom

        def callable_jac_local(params):
            gradients = callable_jac(params) / self.scaling_factor
            self.current_gradients = gradients
            return gradients

        return callable_fom_local, callable_jac_local

    def define_callback(self,plotting_function):
        def callback(*args):
            """ Called at the end of each iteration to record results."""
            self.params_hist.append(copy.copy(self.current_params))
            self.fom_hist.append(self.current_fom)
            self.gradients_hist.append(copy.copy(self.current_gradients))
            self.iteration += 1
            plotting_function()
            self.report_writing()
        self.callback = callback

    def plot(self,fomax,paramsax,gradients_ax):
        fomax.clear()
        paramsax.clear()
        gradients_ax.clear()
        fomax.plot(range(self.iteration),self.fom_hist)
        fomax.set_xlabel('Iteration')
        fomax.set_title('Figure of Merit')
        fomax.set_ylabel('FOM')
        paramsax.semilogy(range(self.iteration),np.abs(self.params_hist))
        paramsax.set_xlabel('Iteration')
        paramsax.set_ylabel('Parameters')
        paramsax.set_title("Parameter evolution")
        gradients_ax.semilogy(range(self.iteration),np.abs(self.gradients_hist))
        gradients_ax.set_xlabel('Iteration')
        gradients_ax.set_ylabel('Gradient Magnitude')
        gradients_ax.set_title("Gradient evolution")

    def report_writing(self):
        with open('optimization_report.txt','a') as f:
            f.write('AT ITERATION {}:  FOM = {}\n'.format(self.iteration,self.fom_hist[-1]))
            f.write('PARAMETERS = {}\n'.format(self.params_hist[-1]/self.scaling_factor))
            f.write('\n \n')