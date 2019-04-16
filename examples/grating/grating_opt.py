""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

######## IMPORTS ########
# General purpose imports
import os
import numpy as np

# Optimization specific imports
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.utilities.wavelengths import Wavelengths
from lumopt.utilities.materials import Material
from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.optimization import Optimization

######## DEFINE BASE SIMULATION ########
script = load_from_lsf(os.path.join(os.path.dirname(__file__), 'grating_base.lsf'))

######## DEFINE SPECTRAL RANGE #########
wavelengths = Wavelengths(start = 1550e-9, stop = 1550e-9, points = 1)

######## DEFINE MATERIALS ##############
oxide = Material(1.44**2, mesh_order = 1)
silicon = Material(3.4**2)

######## DEFINE OPTIMIZABLE GEOMETRY ########
n_grates = 22

def grate_function_generator(grate_number, grate_height = 70e-9, grate_center_y = 220e-9-35e-9):
    def grate_function(params):
        grate_position = params[grate_number]
        grate_width = params[int(grate_number + len(params)/2)]
        left = grate_position - grate_width/2.0
        right = grate_position + grate_width/2.0
        top = grate_center_y + grate_height/2.0
        bottom = grate_center_y - grate_height/2.0
        polygon_points_x = [left, left, right, right]
        polygon_points_y = [top, bottom, bottom, top]
        polygon_points=[(x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
        return np.array(polygon_points)
    return grate_function

initial_positions = 550e-9*np.arange(n_grates)+300e-9
initial_widths = 100e-9*np.ones(n_grates)
edge_precision = 20
bounds = [(250e-9, 15e-6)]*n_grates+[(100e-9,500e-9)]*n_grates
initial_params = np.concatenate((initial_positions,initial_widths))
first_tooth = FunctionDefinedPolygon(func = grate_function_generator(0), initial_params = initial_params, bounds = bounds, z = 0.0, depth = 1e-6, eps_out = silicon, eps_in = oxide, edge_precision = edge_precision, dx = 0.1e-9)
full_geometry = first_tooth
for i in range(1,n_grates):
    new_tooth = FunctionDefinedPolygon(func = grate_function_generator(i), initial_params = initial_params, bounds=bounds, z = 0.0, depth = 1e-6, eps_out = silicon, eps_in = oxide, edge_precision = edge_precision, dx = 0.1e-9)
    full_geometry = full_geometry*new_tooth

######## DEFINE FIGURE OF MERIT ########
fom = ModeMatch(monitor_name = 'fom', mode_number = 1, direction = 'Forward',  multi_freq_src = False, target_T_fwd = lambda wl: np.ones(wl.size), norm_p = 1)

######## DEFINE OPTIMIZATION ALGORITHM ########
optimizer = ScipyOptimizers(max_iter = 30, method = 'L-BFGS-B', scaling_factor = 1e6, pgtol = 1e-9)

######## PUT EVERYTHING TOGETHER ########
opt = Optimization(base_script = script, wavelengths = wavelengths, fom = fom, geometry = full_geometry, optimizer = optimizer, hide_fdtd_cad = False, use_deps = True)

######## RUN THE OPTIMIZER ########
opt.run()

