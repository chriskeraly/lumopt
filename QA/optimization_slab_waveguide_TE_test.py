""" Copyright (c) 2019 Lumerical Inc. """

import sys
sys.path.append(".")
import os
import numpy as np

from qatools import *

from lumopt.utilities.wavelengths import Wavelengths
from lumopt.utilities.materials import Material
from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.optimization import Optimization

class TestOptimizationSlabWaveguideTE(TestCase):
    """ 
        Unit test for the Optimization class. It performs a sanity check that the optimizer converges using a
        simple a slab dielectric waveguide. The waveguide has a gap that must be filled all the way by a polygon
        under optimization to maximize transmission. Unlike the parallel plate waveguide test, this tests uses varFDTD.
    """

    file_dir = os.path.abspath(os.path.dirname(__file__))

    def setUp(self):
        # Base simulation script
        self.base_script = os.path.join(self.file_dir, 'optimization_slab_waveguide_TE_test.lms')
        # Simulation bandwidth        
        self.wavelengths = Wavelengths(start = 1500e-9,
                                       stop = 1600e-9,
                                       points = 11)
        # Polygon defining a rectangle that can grow or shrink along the x-axis to fill the gap
        self.mesh_del = 10.0e-9; # must be kept in sych with self.base_script
        initial_points_x = np.array([self.mesh_del, 1.75 * self.mesh_del])
        def wall(param = initial_points_x):
            assert param.size == 2, "walls defined by two points."
            self.wg_gap = 10.0 * self.mesh_del # must be kept in sych
            points_x = np.array([param[0], param[1], -param[1], -param[0]])
            points_y = 0.5 * np.array([-self.wg_gap, self.wg_gap, self.wg_gap, -self.wg_gap])
            polygon_points = [(x, y) for x, y in zip(points_x, points_y)]
            return np.array(polygon_points)
        self.wg_width = 50.0 * self.mesh_del # must be kept in synch
        bounds = [(0.0, self.wg_width / 2.0)] * initial_points_x.size 
        self.geometry = FunctionDefinedPolygon(func = wall, 
                                               initial_params = initial_points_x, 
                                               bounds = bounds,
                                               z = 0.0, # must be kept in sych
                                               depth = 2.0e-6, # must be kept in sych
                                               eps_out = 1.0 ** 2, # must be kept in sych with
                                               eps_in = Material(base_epsilon = 4.0 ** 2, name = '<Object defined dielectric>', mesh_order = 1), # must be kept in sych with
                                               edge_precision = 5,
                                               dx = 1.0e-10)
        # Figure of merit
        self.fom = ModeMatch(monitor_name = 'fom', # must be kept in sych
                             mode_number = 1, # must be kept in sych
                             direction = 'Backward',
                             multi_freq_src = False,
                             target_T_fwd = lambda wl: np.ones(wl.size),
                             norm_p = 1)
        # Scipy optimizer
        self.optimizer = ScipyOptimizers(max_iter = 5, 
                                         method = 'L-BFGS-B',
                                         scaling_factor = 1.0e6,
                                         pgtol = 1.0e-4,
                                         ftol = 1.0e-12,
                                         target_fom = 0.0,
                                         scale_initial_gradient_to = None)

    def test_permittivity_derivatives(self):
        print("varFDTD optimization with permittivity derivatives (use_deps = True): ")
        opt = Optimization(base_script = self.base_script, 
                           wavelengths = self.wavelengths,
                           fom = self.fom,
                           geometry = self.geometry,
                           optimizer = self.optimizer,
                           use_var_fdtd = True,
                           hide_fdtd_cad = True,
                           use_deps = True,
                           plot_history = False,
                           store_all_simulations = False)
        fom, params = opt.run()
        self.assertGreaterEqual(fom, 0.99991)
        self.assertAlmostEqual(params[0], self.wg_width / 2.0 * self.optimizer.scaling_factor)
        self.assertAlmostEqual(params[1], self.wg_width / 2.0 * self.optimizer.scaling_factor)

    def test_shape_boundary_approximation(self):
        print("varFDTD optimization with shape boundary approximation (use_deps = False): ")
        self.geometry.bounds = [(self.mesh_del, self.wg_width / 2.0 - self.mesh_del)] * len(self.geometry.bounds)
        opt = Optimization(base_script = self.base_script, 
                           wavelengths = self.wavelengths,
                           fom = self.fom,
                           geometry = self.geometry,
                           optimizer = self.optimizer,
                           use_var_fdtd = True,
                           hide_fdtd_cad = True,
                           use_deps = False,
                           plot_history = False,
                           store_all_simulations = False)
        fom, params = opt.run()
        self.assertGreaterEqual(fom, 0.972)
        self.assertAlmostEqual(params[0], (self.wg_width / 2.0 - self.mesh_del) * self.optimizer.scaling_factor)
        self.assertAlmostEqual(params[1], (self.wg_width / 2.0 - self.mesh_del) * self.optimizer.scaling_factor)

if __name__ == "__main__":
    run([__file__])