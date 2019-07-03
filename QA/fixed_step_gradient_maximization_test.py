""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

    
import sys
sys.path.append(".")
import os
import numpy as np

from qatools import *

from lumopt.optimizers.fixed_step_gradient_descent import FixedStepGradientDescent

class FixedStepGradientDescentMaximizationTest(TestCase):
    """ 
        Unit test for the AdaptiveGradientDescent class. 
    """

    machine_eps = np.finfo(float).eps

    def test_single_parameter_maximization(self):
        optimizer = FixedStepGradientDescent(max_dx = 0.005,
                                             max_iter = 99, 
                                             all_params_equal = False,
                                             noise_magnitude = 0.0,
                                             scaling_factor = 1.0)
        fom = lambda param: -1.0/np.sin(np.pi*param[0])
        jac = lambda param: np.pi*np.cos(np.pi*param[0]) / np.power(np.sin(np.pi*param[0]), 2)
        start = np.array(0.005)
        bounds = np.array([(self.machine_eps, 1.0-self.machine_eps)])
        def plot_fun(): pass
        optimizer.initialize(start_params = start, callable_fom = fom, callable_jac = jac, bounds = bounds, plotting_function = plot_fun)
        results = optimizer.run()
        self.assertAlmostEqual(results['x'].size, 1)
        self.assertAlmostEqual(results['x'][0], 0.5, 8)
        self.assertAlmostEqual(results['fun'], -1.0, 3)
        self.assertLessEqual(results['nit'], 99)

if __name__ == "__main__":
    run([__file__])