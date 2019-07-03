""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

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

class TestCoOptimizationParallelPlateWaveguide(TestCase):
    """ 
        Unit test for the Optimization class. It performs a co-optimization using a parallel plate waveguide
        filled by a dielectric excited with two different polarizations (TE and TM). The waveguide has a gap
        that must be filled all the way to maximize transmission.

    """

    file_dir = os.path.abspath(os.path.dirname(__file__))

    def setUp(self):
        # Base simulation project files
        self.base_TE_sim = os.path.join(self.file_dir, 'co_optimization_parallel_plate_waveguide_TE_base.fsp')
        self.base_TM_sim = os.path.join(self.file_dir, 'co_optimization_parallel_plate_waveguide_TM_base.fsp')
        # Simulation bandwidth        
        self.wavelengths = Wavelengths(start = 1500e-9,
                                       stop = 1600e-9,
                                       points = 12)
        # Polygon defining a rectangle that can grow or shrink along the y-axis to fill the gap
        self.mesh_del = 10.0e-9; # must be kept in sych with self.base_script
        initial_points_y = np.array([1.75 * self.mesh_del, 0.01 * self.mesh_del])
        def wall(param = initial_points_y):
            assert param.size == 2, "walls defined by two points."
            self.wg_gap = 10.0 * self.mesh_del # must be kept in sych
            points_x = 0.5 * np.array([-self.wg_gap, self.wg_gap, self.wg_gap, -self.wg_gap])
            points_y = np.array([-param[0], -param[1], param[1], param[0]])
            polygon_points = [(x, y) for x, y in zip(points_x, points_y)]
            return np.array(polygon_points)
        self.wg_width = 50.0 * self.mesh_del # must be kept in synch
        bounds = [(0.0, self.wg_width / 2.0)] * initial_points_y.size 
        self.geometry = FunctionDefinedPolygon(func = wall, 
                                               initial_params = initial_points_y, 
                                               bounds = bounds,
                                               z = 0.0, # must be kept in sych
                                               depth = self.wg_width,
                                               eps_out = Material(base_epsilon = 1.0 ** 2, name = '<Object defined dielectric>', mesh_order = 2), # must be kept in synch
                                               eps_in = Material(base_epsilon = 4.0 ** 2, name = '<Object defined dielectric>', mesh_order = 1), # must be kept in sych
                                               edge_precision = 50,
                                               dx = 1.0e-10)
        # Figure of merit
        self.fom = ModeMatch(monitor_name = 'fom', # must be kept in sych
                             mode_number = 1, # must be kept in sych
                             direction = 'Forward',
                             multi_freq_src = True,
                             target_T_fwd = lambda wl: np.ones(wl.size),
                             norm_p = 1)
        # Scipy optimizer
        self.optimizer = ScipyOptimizers(max_iter = 5, 
                                         method = 'L-BFGS-B',
                                         scaling_factor = 1.0e7,
                                         pgtol = 1.0e-5,
                                         ftol = 1.0e-12,
                                         target_fom = 0.0,
                                         scale_initial_gradient_to = None)

    def test_co_optimization_in_2D(self):
        print("2D TE-TM co-optimization (use_deps = True): ")
        #base_script, wavelengths, fom, geometry, optimizer, use_var_fdtd = False, hide_fdtd_cad = False, use_deps = True, plot_history = True, store_all_simulations = True
        optTE = Optimization(base_script = self.base_TE_sim, 
                             wavelengths = self.wavelengths,
                             fom = self.fom,
                             geometry = self.geometry,
                             optimizer = self.optimizer,
                             use_var_fdtd = False,
                             hide_fdtd_cad = True,
                             use_deps = True,
                             plot_history = False,
                             store_all_simulations = False)
        optTM = Optimization(base_script = self.base_TM_sim, 
                             wavelengths = self.wavelengths,
                             fom = self.fom,
                             geometry = self.geometry,
                             optimizer = self.optimizer,
                             use_var_fdtd = False,
                             hide_fdtd_cad = True,
                             use_deps = True,
                             plot_history = False,
                             store_all_simulations = False)
        opt = optTE + optTM
        fom, params = opt.run()
        self.assertGreaterEqual(fom, 1.99990)
        reference_value = self.wg_width / 2.0 * self.optimizer.scaling_factor[0]
        self.assertAlmostEqual(params[0], reference_value)
        self.assertAlmostEqual(params[1], reference_value)

if __name__ == "__main__":
    run([__file__])
