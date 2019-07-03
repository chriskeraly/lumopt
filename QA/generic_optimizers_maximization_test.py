""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

    
import sys
sys.path.append(".")
import os
import numpy as np

from qatools import *

from lumopt.optimizers.generic_optimizers import ScipyOptimizers

class GenericOptimizersMaximizationTest(TestCase):
    """ 
        Unit test for the ScipyOptimizers class. It checks that the optimizer can indeed maximize a figure of merit  with a known unique maximum 
        within the provided bounds.
    """

    machine_eps = np.finfo(float).eps

    def test_single_parameter_maximization(self):
        optimizer = ScipyOptimizers(max_iter = 200, 
                                    method = 'L-BFGS-B',
                                    scaling_factor = 1.0,
                                    pgtol = 1.0e-11,
                                    ftol = 1.0e-12,
                                    scale_initial_gradient_to = None)
        fom = lambda param: -1.0/np.sin(np.pi*param[0])
        jac = lambda param: np.pi*np.cos(np.pi*param[0])/np.power(np.sin(np.pi*param[0]),2)
        start = np.array(1.0e-6)
        bounds = np.array([(self.machine_eps, 1.0-self.machine_eps)])
        def plot_fun(): pass
        optimizer.initialize(start_params = start, callable_fom = fom, callable_jac = jac, bounds = bounds, plotting_function = plot_fun)
        results = optimizer.run()
        self.assertTrue(results.success)
        self.assertAlmostEqual(results.x.size, 1)
        self.assertAlmostEqual(results.x[0], 0.5, 11)
        self.assertAlmostEqual(results.fun, -1.0, 11)
        self.assertLessEqual(results.nit, 31)
        self.assertLessEqual(results.nfev, 46)

    def test_single_parameter_maximization_with_scaling(self):
        span = 1.0e-9
        optimizer = ScipyOptimizers(max_iter = 200, 
                                    method = 'L-BFGS-B',
                                    scaling_factor = 1.0/span,
                                    pgtol = 1.0e-11)
        fom = lambda param: -1.0/np.sin(np.pi/span*param[0])
        jac = lambda param: np.pi*np.cos(np.pi/span*param[0])/np.power(np.sin(np.pi/span*param[0]),2)/span
        start = np.array(1.0e-6)*span
        bounds = np.array([(self.machine_eps, 1.0-self.machine_eps)])*span
        def plot_fun(): pass
        optimizer.initialize(start_params = start, callable_fom = fom, callable_jac = jac, bounds = bounds, plotting_function = plot_fun)
        results = optimizer.run()
        self.assertTrue(results.success)
        self.assertAlmostEqual(results.x.size, 1)
        self.assertAlmostEqual(results.x[0], 0.5*span, 11)
        self.assertAlmostEqual(results.fun, -1.0, 11)
        self.assertLessEqual(results.nit, 31)
        self.assertLessEqual(results.nfev, 46)

    def test_two_parameter_maximization(self):
        optimizer = ScipyOptimizers(max_iter = 200, 
                                    method = 'L-BFGS-B',
                                    scaling_factor = 1.0,
                                    pgtol = 1.0e-11)
        fom = lambda params: np.sin(np.pi*params[0])*np.sin(np.pi*params[1])
        jac = lambda params: np.pi*np.array([np.cos(np.pi*params[0])*np.sin(np.pi*params[1]),
                                             np.sin(np.pi*params[0])*np.cos(np.pi*params[1])])
        start = np.array([1.0e-5, 1.0-1e-5])
        bounds = np.array([(self.machine_eps, 1.0-self.machine_eps), (self.machine_eps, 1.0-self.machine_eps)])
        def plot_fun(): pass
        optimizer.initialize(start_params = start, callable_fom = fom, callable_jac = jac, bounds = bounds, plotting_function = plot_fun)
        results = optimizer.run()
        self.assertTrue(results.success)
        self.assertAlmostEqual(results.x.size, 2)
        self.assertAlmostEqual(results.x[0], 0.5, 11)
        self.assertAlmostEqual(results.x[1], 0.5, 11)
        self.assertAlmostEqual(results.fun, 1.0, 11)
        self.assertLessEqual(results.nit, 3)
        self.assertLessEqual(results.nfev, 15)
    
    def test_two_parameter_maximization_with_scaling(self):
        span = 1.0e-9
        optimizer = ScipyOptimizers(max_iter = 200, 
                                    method = 'L-BFGS-B',
                                    scaling_factor = np.array([1.0/span, 1.0]),
                                    pgtol = 1.0e-11)
        fom = lambda params: np.sin(np.pi/span*params[0])*np.sin(np.pi*params[1])
        jac = lambda params: np.pi*np.array([np.cos(np.pi/span*params[0])*np.sin(np.pi*params[1])/span,
                                             np.sin(np.pi/span*params[0])*np.cos(np.pi*params[1])])
        start = np.array([1.0e-5*span, 1.0-1e-5])
        bounds = np.array([(self.machine_eps*span, (1.0-self.machine_eps)*span), (self.machine_eps, 1.0-self.machine_eps)])
        def plot_fun(): pass
        optimizer.initialize(start_params = start, callable_fom = fom, callable_jac = jac, bounds = bounds, plotting_function = plot_fun)
        results = optimizer.run()
        self.assertTrue(results.success)
        self.assertAlmostEqual(results.x.size, 2)
        self.assertAlmostEqual(results.x[0], 0.5*span, 11)
        self.assertAlmostEqual(results.x[1], 0.5, 11)
        self.assertAlmostEqual(results.fun, 1.0, 11)
        self.assertLessEqual(results.nit, 3)
        self.assertLessEqual(results.nfev, 15)

if __name__ == "__main__":
    run([__file__])