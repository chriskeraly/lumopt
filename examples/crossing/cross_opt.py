""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

######## IMPORTS ########
import os
import numpy as np
import scipy as sp
from lumopt import CONFIG

from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt.utilities.materials import Material
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.optimization import Optimization

######## DEFINE BASE SIMULATION ########
crossing_base = load_from_lsf(os.path.join(CONFIG['root'], 'examples/crossing/crossing_base_TE_modematch_2D.lsf'))

######## DEFINE OPTIMIZABLE GEOMETRY ########
def cross(params):
    y_end = params[-1]
    x_end = 0 - y_end
    points_x = np.concatenate(([-2.01e-6], np.linspace(-2e-6, x_end, 10)))
    points_y = np.concatenate(([0.25e-6], params))
    n_interpolation_points = 50
    polygon_points_x = np.linspace(min(points_x), max(points_x), n_interpolation_points)
    interpolator = sp.interpolate.interp1d(points_x, points_y, kind = 'cubic')
    polygon_points_y = [max(min(point, 1e-6), -1e-6) for point in interpolator(polygon_points_x)]
    pplu = [(x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
    ppld = [(x, -y) for x, y in zip(polygon_points_x, polygon_points_y)]
    ppdl = [(-y, x) for x, y in zip(polygon_points_x, polygon_points_y)]
    ppdr = [(y, x) for x, y in zip(polygon_points_x, polygon_points_y)]
    pprd = [(-x, -y) for x, y in zip(polygon_points_x, polygon_points_y)]
    ppru = [(-x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
    ppur = [(y, -x) for x, y in zip(polygon_points_x, polygon_points_y)]
    ppul = [(-y, -x) for x, y in zip(polygon_points_x, polygon_points_y)]
    polygon_points = np.array(pplu[::-1] + ppld[:-1] + ppdl[::-1] + ppdr[:-1] + pprd[::-1] + ppru[:-1] + ppur[::-1] + ppul[:-1])
    return polygon_points

polygon_geometry = FunctionDefinedPolygon(func = cross, initial_params = np.linspace(0.25e-6, 0.6e-6, 10), bounds = [(0.2e-6, 1e-6)]*10, z = 0.0, depth = 220.0e-9, eps_out = Material(1.44 ** 2), eps_in=Material(2.8 ** 2, 2), edge_precision = 5, dx = 0.01e-9)

######## DEFINE FIGURE OF MERIT ########
mode_fom = ModeMatch(monitor_name = 'fom', wavelengths = 1550e-9, mode_number = 2, direction = 'Forward')

######## DEFINE OPTIMIZATION ALGORITHM ########
scipy_optimizer = ScipyOptimizers(max_iter=20)

######## PUT EVERYTHING TOGETHER ########
opt = Optimization(base_script = crossing_base, fom = mode_fom, geometry = polygon_geometry, optimizer = scipy_optimizer)

######## RUN THE OPTIMIZER ########
opt.run()
