""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

######## IMPORTS ########
# General purpose imports
import os
import numpy as np
import scipy as sp

from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.utilities.wavelengths import Wavelengths
from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.optimization import Optimization

######## DEFINE SPECTRAL RANGE #########
wavelengths = Wavelengths(start = 1550e-9, stop = 1550e-9, points = 1)

######## DEFINE BASE SIMULATION ########
# Use the same script for both simulations, but it's just to keep the example simple. You could use two.
script_1 = load_from_lsf(os.path.join(os.path.dirname(__file__), 'splitter_base_TE_modematch.lsf'))
script_2 = load_from_lsf(os.path.join(os.path.dirname(__file__), 'splitter_base_TE_modematch_25nmoffset.lsf'))

######## DEFINE OPTIMIZABLE GEOMETRY ########

## Here the two splitters just have a 25nm offset from each other, so that the result is robust
def taper_splitter_1(params,n_points=10):
    points_x = np.concatenate(([-1.01e-6],np.linspace(-1.1e-6,0.9e-6,10),[1.01e-6]))
    points_y = np.concatenate(([0.25e-6],params,[0.6e-6]))
    n_interpolation_points=100
    polygon_points_x = np.linspace(min(points_x), max(points_x), n_interpolation_points)
    interpolator = sp.interpolate.interp1d(points_x, points_y, kind='cubic')
    polygon_points_y = [max(min(point,1e-6),-1e-6) for point in interpolator(polygon_points_x)]

    polygon_points_up = [(x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
    polygon_points_down = [(x, -y) for x, y in zip(polygon_points_x, polygon_points_y)]
    polygon_points = np.array(polygon_points_up[::-1] + polygon_points_down)
    return polygon_points

dx=25e-9
def taper_splitter_2(params,n_points=10):
    points_x = np.concatenate(([-1.01e-6],np.linspace(-1.1e-6,0.9e-6,10),[1.01e-6]))
    points_y = np.concatenate(([0.25e-6+dx],params+dx,[0.6e-6+dx]))
    n_interpolation_points=100
    polygon_points_x = np.linspace(min(points_x), max(points_x), n_interpolation_points)
    interpolator = sp.interpolate.interp1d(points_x, points_y, kind='cubic')
    polygon_points_y = [max(min(point,1e-6),-1e-6) for point in interpolator(polygon_points_x)]

    polygon_points_up = [(x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
    polygon_points_down = [(x, -y) for x, y in zip(polygon_points_x, polygon_points_y)]
    polygon_points = np.array(polygon_points_up[::-1] + polygon_points_down)
    return polygon_points

bounds = [(0.2e-6, 1e-6)]*10
# final value from splitter_opt_2D.py optimization
initial_params = np.array([2.44788514e-07, 2.65915795e-07, 2.68748023e-07, 4.42233947e-07, 6.61232152e-07, 6.47561406e-07, 6.91473099e-07, 6.17511522e-07, 6.70669074e-07, 5.86141086e-07])
geometry_1 = FunctionDefinedPolygon(func = taper_splitter_1, initial_params = initial_params, bounds = bounds, z = 0.0, depth = 220e-9, eps_out = 1.44 ** 2, eps_in = 2.8 ** 2, edge_precision = 5, dx = 0.1e-9)
geometry_2 = FunctionDefinedPolygon(func = taper_splitter_2, initial_params = initial_params, bounds = bounds, z = 0.0, depth = 220e-9, eps_out = 1.44 ** 2, eps_in = 2.8 ** 2, edge_precision = 5, dx = 0.1e-9)

######## DEFINE FIGURE OF MERIT ########
# Although we are optimizing for the same thing, two separate fom objects must be create
fom_1 = ModeMatch(monitor_name = 'fom', mode_number = 3, direction = 'Forward')
fom_2 = ModeMatch(monitor_name = 'fom', mode_number = 3, direction = 'Forward')

######## DEFINE OPTIMIZATION ALGORITHM ########
#For the optimizer, they should all be set the same, but different objects. Eventually this will be improved
optimizer_1 = ScipyOptimizers(max_iter = 40)
optimizer_2 = ScipyOptimizers(max_iter = 40)

######## PUT EVERYTHING TOGETHER ########
opt_1 = Optimization(base_script = script_1, wavelengths = wavelengths, fom = fom_1, geometry = geometry_1, optimizer = optimizer_1)
opt_2 = Optimization(base_script = script_2, wavelengths = wavelengths, fom = fom_2, geometry = geometry_2, optimizer = optimizer_2)
opt = opt_1 + opt_2

######## RUN THE OPTIMIZER ########
opt.run()