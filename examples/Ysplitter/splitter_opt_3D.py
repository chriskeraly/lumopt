""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

######## IMPORTS ########
# General purpose imports
import os
import numpy as np
import scipy as sp

# Optimization specific imports
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.utilities.wavelengths import Wavelengths
from lumopt.utilities.materials import Material
from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.optimization import Optimization

######## DEFINE BASE SIMULATION ########
script = load_from_lsf(os.path.join(os.path.dirname(__file__), 'splitter_base_TE_modematch_3D.lsf'))

######## DEFINE SPECTRAL RANGE #########
wavelengths = Wavelengths(start = 1300e-9, stop = 1800e-9, points = 21)

######## DEFINE DATABASE MATERIALS #####
silicon = Material(name = 'Si (Silicon) - Palik', mesh_order = 2)

######## DEFINE OPTIMIZABLE GEOMETRY ########
initial_points_x = np.linspace(-1.0e-6,1.0e-6, 10)
initial_points_y = 0.25e-6 + (0.6e-6 - 0.25e-6) * np.power(np.sin( np.pi / 2.0 * (initial_points_x - initial_points_x.min()) / (initial_points_x.max() - initial_points_x.min()) ), 2)
def taper_splitter(params = initial_points_y):
    ''' Defines a taper where the parameters are the y coordinates of the nodes of a cubic spline.'''
    points_x = np.concatenate(([initial_points_x.min() - 0.01e-6], initial_points_x, [initial_points_x.max() + 0.01e-6]))
    points_y = np.concatenate(([initial_points_y.min()], params, [initial_points_y.max()]))
    n_interpolation_points = 100
    polygon_points_x = np.linspace(min(points_x), max(points_x), n_interpolation_points)
    interpolator = sp.interpolate.interp1d(points_x, points_y, kind = 'cubic')
    polygon_points_y = interpolator(polygon_points_x)
    polygon_points_up = [(x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
    polygon_points_down = [(x, -y) for x, y in zip(polygon_points_x, polygon_points_y)]
    polygon_points = np.array(polygon_points_up[::-1] + polygon_points_down)
    return polygon_points
bounds = [(0.2e-6, 0.8e-6)] * initial_points_y.size
geometry = FunctionDefinedPolygon(func = taper_splitter, initial_params = initial_points_y, bounds = bounds, z = 0.0, depth = 220e-9, eps_out = 1.44 ** 2, eps_in = silicon, edge_precision = 5, dx = 0.1e-9)

######## DEFINE FIGURE OF MERIT ########
fom = ModeMatch(monitor_name = 'fom', mode_number = 1, direction = 'Forward', multi_freq_src = False, target_T_fwd = lambda wl: np.ones(wl.size), norm_p = 1)

######## DEFINE OPTIMIZATION ALGORITHM ########
optimizer = ScipyOptimizers(max_iter = 20, method = 'L-BFGS-B', scaling_factor = 1e6, pgtol = 1e-9)

######## PUT EVERYTHING TOGETHER ########
opt = Optimization(base_script = script, wavelengths = wavelengths, fom = fom, geometry = geometry, optimizer = optimizer, hide_fdtd_cad = False, use_deps = True)

######## RUN THE OPTIMIZER ########
opt.run()
