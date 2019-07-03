""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

    
import sys
sys.path.append(".")
import os
import numpy as np

from qatools import *

from lumopt.optimizers.adaptive_gradient_descent import AdaptiveGradientDescent

class AdaptiveGradientDescentMaximizationTest(TestCase):
    """ 
        Unit test for the AdaptiveGradientDescent class. It performs a quick sanity check that the optimizer can indeed maximize a figure of merit
        with a known unique maximum within the provided bounds.
    """

    machine_eps = np.finfo(float).eps

    def test_single_parameter_maximization(self):
        optimizer = AdaptiveGradientDescent(max_dx = 0.5,
                                            min_dx = 1.0e-12, 
                                            max_iter = 7, 
                                            dx_regrowth_factor = 2.0,
                                            all_params_equal = True,
                                            scaling_factor = 1.0)
        fom = lambda param: -1.0/np.sin(np.pi*param)
        jac = lambda param: np.pi*np.cos(np.pi*param) / np.power(np.sin(np.pi*param), 2)
        start = np.array([1.0e-6])
        bounds = np.array([(self.machine_eps, 1.0-self.machine_eps)])
        def plot_fun(): pass
        optimizer.initialize(start_params = start, callable_fom = fom, callable_jac = jac, bounds = bounds, plotting_function = plot_fun)
        results = optimizer.run()
        self.assertAlmostEqual(results['x'].size, 1)
        self.assertAlmostEqual(results['x'][0], 0.5, 8)
        self.assertAlmostEqual(results['fun'], -1.0, 8)
        self.assertLessEqual(results['nit'], 7)

    def test_single_parameter_maximization_with_scaling(self):
        span = 1.0e-9
        optimizer = AdaptiveGradientDescent(max_dx = 0.5*span,
                                           min_dx = 1.0e-12*span,
                                           max_iter = 7, 
                                           dx_regrowth_factor = 2.0,
                                           all_params_equal = True,
                                           scaling_factor = 1.0/span)
        fom = lambda param: -1.0/np.sin(np.pi/span*param[0])
        jac = lambda param: np.pi*np.cos(np.pi/span*param[0])/np.power(np.sin(np.pi/span*param[0]),2)/span
        start = np.array(1.0e-6)*span
        bounds = np.array([(self.machine_eps, 1.0-self.machine_eps)])*span
        def plot_fun(): pass
        optimizer.initialize(start_params = start, callable_fom = fom, callable_jac = jac, bounds = bounds, plotting_function = plot_fun)
        results = optimizer.run()
        self.assertAlmostEqual(results['x'].size, 1)
        self.assertAlmostEqual(results['x'][0], 0.5*span, 8)
        self.assertAlmostEqual(results['fun'], -1.0, 8)
        self.assertLessEqual(results['nit'], 7)

    def test_two_parameter_maximization(self):
        optimizer = AdaptiveGradientDescent(max_dx = 0.5*np.ones(2),
                                            min_dx = 1.0e-12*np.ones(2),
                                            max_iter = 8, 
                                            dx_regrowth_factor = 2.0,
                                            all_params_equal = False,
                                            scaling_factor = np.ones(2))
        fom = lambda params: np.sin(np.pi*params[0])*np.sin(np.pi*params[1])
        jac = lambda params: np.pi*np.array([np.cos(np.pi*params[0])*np.sin(np.pi*params[1]),
                                             np.sin(np.pi*params[0])*np.cos(np.pi*params[1])])
        start = np.array([1.0e-5, 1.0-1e-5])
        bounds = np.array([(self.machine_eps, 1.0-self.machine_eps), (self.machine_eps, 1.0-self.machine_eps)])
        def plot_fun(): pass
        optimizer.initialize(start_params = start, callable_fom = fom, callable_jac = jac, bounds = bounds, plotting_function = plot_fun)
        results = optimizer.run()
        self.assertAlmostEqual(results['x'].size, 2)
        self.assertAlmostEqual(results['x'][0], 0.5, 8)
        self.assertAlmostEqual(results['x'][1], 0.5, 8)
        self.assertAlmostEqual(results['fun'], 1.0, 8)
        self.assertLessEqual(results['nit'], 8)

    def test_two_parameter_maximization(self):
        span = 1.0e-6
        optimizer = AdaptiveGradientDescent(max_dx = np.array([0.5*span, 0.5]),
                                            min_dx = np.array([1.0e-12*span, 1.0e-12]),
                                            max_iter = 8, 
                                            dx_regrowth_factor = 2.0,
                                            all_params_equal = False,
                                            scaling_factor = np.array([1.0/span, 1.0]))
        fom = lambda params: np.sin(np.pi/span*params[0])*np.sin(np.pi*params[1])
        jac = lambda params: np.pi*np.array([np.cos(np.pi/span*params[0])*np.sin(np.pi*params[1])/span,
                                             np.sin(np.pi/span*params[0])*np.cos(np.pi*params[1])])
        start = np.array([1.0e-5*span, 1.0-1e-5])
        bounds = np.array([(self.machine_eps*span, (1.0-self.machine_eps)*span), (self.machine_eps, 1.0-self.machine_eps)])
        def plot_fun(): pass
        optimizer.initialize(start_params = start, callable_fom = fom, callable_jac = jac, bounds = bounds, plotting_function = plot_fun)
        results = optimizer.run()
        self.assertAlmostEqual(results['x'].size, 2)
        self.assertAlmostEqual(results['x'][0], 0.5*span, 8)
        self.assertAlmostEqual(results['x'][1], 0.5, 8)
        self.assertAlmostEqual(results['fun'], 1.0, 8)
        self.assertLessEqual(results['nit'], 8)

if __name__ == "__main__":
    run([__file__])