""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import sys
sys.path.append(".")
import os
import numpy as np

from qatools import *

from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.utilities.wavelengths import Wavelengths
from lumopt.utilities.materials import Material
from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.optimization import Optimization

class TestOptimizationWaveguideFilterTM2D(TestCase):
    """ 
        Unit test for the Optimization class. It performs a sanity check that the optimizer converges using a
        simple Bragg filter. The width of the two gaps must be optimized to maximize transmission. The optimization
        is done using both single frequency and broadband simulations as well as the two method for computing the
        FOM gradient.
    """

    file_dir = os.path.abspath(os.path.dirname(__file__))

    def setUp(self):
        # Base simulation script
        self.base_script = load_from_lsf(os.path.join(self.file_dir, 'optimization_waveguide_filter_TM_2D_base.lsf'))
        # Simulation bandwidth
        self.wavelengths = Wavelengths(start = 1300e-9,
                                       stop = 1800e-9,
                                       points = 41)
                                       
        # Polygons to form the two gaps
        self.mesh_del = 20.0e-9 # must be kept in sych with self.base_script
        initial_param = 10.0 * np.array([self.mesh_del])
        def rectangle(param = initial_param, offset = 0.0):
            assert param.size == 1, "rectangle grows along a single dimension."
            wg_width = 35.0 * self.mesh_del # must be kept in synch
            points_x = 0.5 * np.array([-wg_width,  wg_width, wg_width, -wg_width])
            points_y = 0.5 * np.array([-param, -param, param,  param]) + offset
            polygon_points = [(x, y) for x, y in zip(points_x, points_y)]
            return np.array(polygon_points)
        bounds = [(self.mesh_del, 20.0 * self.mesh_del)]
        z = 0.0 # must be kept in sych
        depth = 200.0 * self.mesh_del # must be kept in sych
        eps_in = Material(base_epsilon = 1.44 ** 2, mesh_order = 1) # must be kept in sych with
        eps_out = Material(base_epsilon = 2.8 ** 2, mesh_order = 1) # must be kept in sych with
        edge_precision = 25
        dx = 1.0e-10
        self.geometry = (FunctionDefinedPolygon(func = lambda param: rectangle(param[0], 2.0 * param[0]), initial_params = initial_param, bounds = bounds, z = z, depth = depth, eps_out = eps_out, eps_in = eps_in, edge_precision = edge_precision, dx = dx) *
                         FunctionDefinedPolygon(func = lambda param: rectangle(param[0],-2.0 * param[0]), initial_params = initial_param, bounds = bounds, z = z, depth = depth, eps_out = eps_out, eps_in = eps_in, edge_precision = edge_precision, dx = dx))
        # Broadband figure of merit
        target_T_fwd = lambda wl: 0.3 + 0.65*np.power(np.sin(np.pi * (wl - wl.min()) / (wl.max() - wl.min())), 6)
        self.fom = ModeMatch(monitor_name = 'FOM', # must be kept in sych
                             mode_number = 1, # must be kept in sych
                             direction = 'Backward',
                             multi_freq_src = True,
                             target_T_fwd = target_T_fwd,
                             norm_p = 1)
        # Scipy optimzier
        self.optimizer = ScipyOptimizers(max_iter = 10, 
                                         method = 'L-BFGS-B',
                                         scaling_factor = 1.0e7,
                                         pgtol = 1.1e-2,
                                         ftol = 1.0e-12,
                                         target_fom = 0.0,
                                         scale_initial_gradient_to = None)

    def test_broadband_optimization(self):
        print("Broadband optimization results (use_deps = True):")
        opt = Optimization(base_script = self.base_script, 
                           wavelengths = self.wavelengths,
                           fom = self.fom,
                           geometry = self.geometry,
                           optimizer = self.optimizer,
                           use_var_fdtd = False,
                           hide_fdtd_cad = True,
                           use_deps = True,
                           plot_history = False,
                           store_all_simulations = False)
        fom, params = opt.run()
        self.assertAlmostEqual(params[0], 2.050375e-7 * self.optimizer.scaling_factor[0], 4)
        self.assertGreaterEqual(fom, 0.461815)

    def test_broadband_legacy_optimization(self):
        print("Broadband optimization results (use_deps = False):")
        self.optimizer.scaling_factor = np.array(2.0e7)
        opt = Optimization(base_script = self.base_script, 
                           wavelengths = self.wavelengths,
                           fom = self.fom,
                           geometry = self.geometry,
                           optimizer = self.optimizer,
                           use_var_fdtd = False,
                           hide_fdtd_cad = True,
                           use_deps = False,
                           plot_history = False,
                           store_all_simulations = False)
        fom, params = opt.run()
        self.assertAlmostEqual(params[0], 2.05061e-7 * self.optimizer.scaling_factor, 4)
        self.assertGreaterEqual(fom, 0.461815)

    def test_single_wavelength_optimization(self):
        print("Single wavelength optimization results (use_deps = True):")
        self.fom.target_T_fwd = lambda wl: np.ones(wl.size)
        self.fom.multi_freq_src = False
        self.wavelengths = 1550.0e-9
        self.optimizer.scaling_factor = np.array(2.0e7)
        self.optimizer.pgtol = 3.1e-2
        opt = Optimization(base_script = self.base_script, 
                           wavelengths = self.wavelengths,
                           fom = self.fom,
                           geometry = self.geometry,
                           optimizer = self.optimizer,
                           use_var_fdtd = False,
                           hide_fdtd_cad = True,
                           use_deps = True,
                           plot_history = False,
                           store_all_simulations = False)
        fom, params = opt.run()
        self.assertAlmostEqual(params[0], 2.0582452415e-7 * self.optimizer.scaling_factor, 4)
        self.assertGreaterEqual(fom, 0.9192)

    def test_single_wavelength_legacy_optimization(self):
        print("Single wavelength optimization results (use_deps = False):")
        self.fom.target_T_fwd = lambda wl: np.ones(wl.size)
        self.fom.multi_freq_src = False
        self.wavelengths = 1550.0e-9
        self.optimizer.scaling_factor = np.array(2.0e7)
        self.optimizer.pgtol = 3.1e-2
        opt = Optimization(base_script = self.base_script, 
                           wavelengths = self.wavelengths,
                           fom = self.fom,
                           geometry = self.geometry,
                           optimizer = self.optimizer,
                           use_var_fdtd = False,
                           hide_fdtd_cad = True,
                           use_deps = False,plot_history = False,
                           store_all_simulations = False)
        fom, params = opt.run()
        self.assertAlmostEqual(params[0], 2.05609116e-7 * self.optimizer.scaling_factor, 4)
        self.assertGreaterEqual(fom, 0.91905)

if __name__ == "__main__":
    run([__file__])