""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

######## IMPORTS ########
# General purpose imports
import os
import numpy as np
import scipy as sp
from lumopt import CONFIG

from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.optimization import Optimization

######## DEFINE BASE SIMULATION ########
# Use the same script for both simulations, but it's just to keep the example simple. You could use two.
script_1550 = load_from_lsf(os.path.join(CONFIG['root'], 'examples/WDM_splitter/WDM_splitter_base_TE_1550.lsf'))
script_1310 = load_from_lsf(os.path.join(CONFIG['root'], 'examples/WDM_splitter/WDM_splitter_base_TE_1550.lsf')).replace('1550e-9','1310e-9')

######## DEFINE OPTIMIZABLE GEOMETRY ########
separation = 500.0e-9
size_x = 10.0e-6

def lower_coupler_arm(params, n_points = 10):
    points_x = np.concatenate(([0.5e-6], np.linspace(0.55e-6,size_x-0.55e-6,20), [size_x-0.5e-6]))
    points_y = np.concatenate(([-separation/2], params-separation/2, params[::-1]-separation/2, [-separation/2]))
    n_interpolation_points=100
    polygon_points_x = np.linspace(min(points_x), max(points_x), n_interpolation_points)
    interpolator = sp.interpolate.interp1d(points_x, points_y, kind = 'cubic')
    polygon_points_y = [max(min(point,0e-6),-separation/2-0.25e-6) for point in interpolator(polygon_points_x)]
    polygon_points_up = [(x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
    polygon_points_down = [(x, y-0.5e-6) for x, y in zip(polygon_points_x, polygon_points_y)]
    polygon_points = np.array(polygon_points_up[::-1] + polygon_points_down)
    return polygon_points

def upper_coupler_arm(params, n_points = 10):
    points_x = np.concatenate(([0.5e-6], np.linspace(0.55e-6,size_x-0.55e-6,20), [size_x-0.5e-6]))
    points_y = np.concatenate(([separation/2], -params+separation/2,-params[::-1]+separation/2, [separation/2]))
    n_interpolation_points=100
    polygon_points_x = np.linspace(min(points_x), max(points_x), n_interpolation_points)
    interpolator = sp.interpolate.interp1d(points_x, points_y, kind = 'cubic')
    polygon_points_y = [max(min(point,separation/2+0.5e-6),-0e-6) for point in interpolator(polygon_points_x)]
    polygon_points_up = [(x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
    polygon_points_down = [(x, y+0.5e-6) for x, y in zip(polygon_points_x, polygon_points_y)]
    polygon_points = np.array(polygon_points_up + polygon_points_down[::-1])
    return polygon_points

bounds = [(-0.25e-6, 0.25e-6)]*10
initial_params = np.linspace(0,0.24e-6,10)
geometry_1550_lower = FunctionDefinedPolygon(func = lower_coupler_arm, initial_params = initial_params, bounds = bounds, z = 0.0, depth = 220e-9, eps_out = 1.44 ** 2, eps_in = 2.8 ** 2, edge_precision = 5, dx = 0.01e-9)
geometry_1550_upper = FunctionDefinedPolygon(func = upper_coupler_arm, initial_params = initial_params, bounds = bounds, z = 0.0, depth = 220e-9, eps_out = 1.44 ** 2, eps_in = 2.8 ** 2, edge_precision = 5, dx = 0.01e-9)
geometry_1550 = geometry_1550_lower * geometry_1550_upper
geometry_1310_lower = FunctionDefinedPolygon(func = lower_coupler_arm, initial_params = initial_params, bounds = bounds, z = 0.0, depth = 220e-9, eps_out = 1.44 ** 2, eps_in = 2.8 ** 2, edge_precision = 5, dx = 0.01e-9)
geometry_1310_upper = FunctionDefinedPolygon(func = upper_coupler_arm, initial_params = initial_params, bounds = bounds, z = 0.0, depth = 220e-9, eps_out = 1.44 ** 2, eps_in = 2.8 ** 2, edge_precision = 5, dx = 0.01e-9)
geometry_1310 = geometry_1310_lower * geometry_1310_upper

######## DEFINE FIGURE OF MERIT ########
# Although we are optimizing for the same thing, two separate fom objects must be created.

fom_1550 = ModeMatch(monitor_name = 'fom_1550', wavelengths = 1550e-9, mode_number = 2, direction = 'Forward')
fom_1310 = ModeMatch(monitor_name = 'fom_1310', wavelengths = 1310e-9, mode_number = 2, direction = 'Forward')

######## DEFINE OPTIMIZATION ALGORITHM ########
#For the optimizer, they should all be set the same, but different objects. Eventually this will be improved
optimizer_1550 = ScipyOptimizers(max_iter = 40)
optimizer_1310 = ScipyOptimizers(max_iter = 40)

######## PUT EVERYTHING TOGETHER ########
opt_1550 = Optimization(base_script = script_1550, fom = fom_1550, geometry = geometry_1550, optimizer = optimizer_1550)
opt_1310 = Optimization(base_script = script_1310, fom = fom_1310, geometry = geometry_1310, optimizer = optimizer_1310)
opt = opt_1550 + opt_1310

######## RUN THE OPTIMIZER ########
opt.run()
