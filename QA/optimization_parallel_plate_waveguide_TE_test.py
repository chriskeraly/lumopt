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

class TestOptimizationParallelPlateWaveguideTE(TestCase):
    """ 
        Unit test for the Optimization class. It performs a sanity check that the optimizer converges using a
        simple a parallel plate waveguide filled by a dielectric. The waveguide has a gap that must be filled
        all the way by the polygon under optimization to maximize transmission.

        There are two independent methods in the code base to compute the gradient of the figure of merit:
            1) using the permittivity derivatives calculated directly from meshing (use_deps == True) and
            2) using the shape derivative approximation described in Owen Miller's thesis (use_deps == False).
        Both methods are tested here using exactly the same structure.
    """

    file_dir = os.path.abspath(os.path.dirname(__file__))

    def setUp(self):
        # Base simulation script
        self.base_script = load_from_lsf(os.path.join(self.file_dir, 'optimization_parallel_plate_waveguide_TE_base.lsf'))
        # Simulation bandwidth        
        self.wavelengths = Wavelengths(start = 1500e-9,
                                       stop = 1600e-9,
                                       points = 11)
        # Polygon defining a rectangle that can grow or shrink along the y-axis to fill the gap
        self.mesh_del = 10.0e-9; # must be kept in sych with self.base_script
        initial_points_y = np.array([0.01 * self.mesh_del, 1.75 * self.mesh_del])
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
                                               depth = self.wg_width, # must be kept in sych
                                               eps_out = 1.0 ** 2, # must be kept in sych with
                                               eps_in = Material(base_epsilon = 4.0 ** 2, name = '<Object defined dielectric>', mesh_order = 1), # must be kept in sych with
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
                                         pgtol = 1.0e-5)

    def test_permittivity_derivatives_in_2D(self):
        print("2D optimization with permittivity derivatives (use_deps = True): ")
        opt = Optimization(base_script = self.base_script + "setnamed('FDTD','dimension','2D');", 
                           wavelengths = self.wavelengths,
                           fom = self.fom,
                           geometry = self.geometry,
                           optimizer = self.optimizer,
                           hide_fdtd_cad = True,
                           use_deps = True)
        fom, params = opt.run()
        self.assertGreaterEqual(fom, 0.99991)
        self.assertAlmostEqual(params[0], self.wg_width / 2.0 * self.optimizer.scaling_factor)
        self.assertAlmostEqual(params[1], self.wg_width / 2.0 * self.optimizer.scaling_factor)

    def test_permittivity_derivatives_in_3D(self):
        print("3D optimization with permittivity derivatives (use_deps = True): ")
        opt = Optimization(base_script = self.base_script + "setnamed('FDTD','dimension','3D');", 
                           wavelengths = self.wavelengths,
                           fom = self.fom,
                           geometry = self.geometry,
                           optimizer = self.optimizer,
                           hide_fdtd_cad = True,
                           use_deps = True)
        fom, params = opt.run()
        self.assertGreaterEqual(fom, 0.99991)
        self.assertAlmostEqual(params[0], self.wg_width / 2.0 * self.optimizer.scaling_factor)
        self.assertAlmostEqual(params[1], self.wg_width / 2.0 * self.optimizer.scaling_factor)

    def test_shape_boundary_approximation_in_2D(self):
        print("2D optimization with shape boundary approximation (use_deps = False): ")
        self.geometry.bounds = [(0.0, self.wg_width / 2.0 - self.mesh_del)] * len(self.geometry.bounds)
        # Note: bounds are tweaked since the shape boundary approximation method does not work
        #       when the shape under optimization touches the boundary of the FDTD region.
        opt = Optimization(base_script = self.base_script + "setnamed('FDTD','dimension','2D');", 
                           wavelengths = self.wavelengths,
                           fom = self.fom,
                           geometry = self.geometry,
                           optimizer = self.optimizer,
                           hide_fdtd_cad = True,
                           use_deps = False)
        fom, params = opt.run()
        self.assertGreaterEqual(fom, 0.972)
        self.assertAlmostEqual(params[0], (self.wg_width / 2.0 - self.mesh_del) * self.optimizer.scaling_factor)
        self.assertAlmostEqual(params[1], (self.wg_width / 2.0 - self.mesh_del) * self.optimizer.scaling_factor)

    def test_shape_boundary_approximation_in_3D(self):
        print("3D optimization with shape boundary approximation (use_deps = False): ")
        self.geometry.bounds = [(0.0, self.wg_width / 2.0 - self.mesh_del)] * len(self.geometry.bounds)
        # Note: bounds are tweaked since the shape boundary approximation method does not work
        #       when the shape under optimization touches the boundary of the FDTD region.
        opt = Optimization(base_script = self.base_script + "setnamed('FDTD','dimension','3D');", 
                           wavelengths = self.wavelengths,
                           fom = self.fom,
                           geometry = self.geometry,
                           optimizer = self.optimizer,
                           hide_fdtd_cad = True,
                           use_deps = False)
        fom, params = opt.run()
        self.assertGreaterEqual(fom, 0.972)
        self.assertAlmostEqual(params[0], (self.wg_width / 2.0 - self.mesh_del) * self.optimizer.scaling_factor)
        self.assertAlmostEqual(params[1], (self.wg_width / 2.0 - self.mesh_del) * self.optimizer.scaling_factor)

if __name__ == "__main__":
    run([__file__])