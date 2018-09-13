######## IMPORTS ########
# General purpose imports
import numpy as np
import os
import scipy
from lumopt import CONFIG

# Optimization specific imports
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.geometries.polygon import function_defined_Polygon
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimization import Optimization
from lumopt.optimizers.generic_optimizers import ScipyOptimizers

######## DEFINE BASE SIMULATION ########

script = load_from_lsf(os.path.join(CONFIG['root'], 'examples/Ysplitter/splitter_base_TE_modematch_3D.lsf'))

######## DEFINE OPTIMIZABLE GEOMETRY ########

# The class function_defined_Polygon needs a parameterized Polygon (with points ordered
# in a counter-clockwise direction. Here the geometry is defined by 10 parameters defining
# the knots of a spline, and the resulting Polygon has 200 edges, making it quite smooth

def taper_splitter(params=np.linspace(0.25e-6, 0.6e-6, 10)):
    '''Just a taper where the parameters are the y coordinates of the nodes of a cubic spline'''
    points_x=np.concatenate(([-1.01e-6],np.linspace(-1e-6,1e-6,10),[1.01e-6]))
    points_y=np.concatenate(([0.25e-6],params,[0.6e-6]))

    n_interpolation_points=100
    polygon_points_x = np.linspace(min(points_x), max(points_x), n_interpolation_points)
    interpolator = scipy.interpolate.interp1d(points_x, points_y, kind='cubic')
    polygon_points_y = interpolator(polygon_points_x)
    polygon_points_y=np.maximum(0.2e-6,(np.minimum(1e-6,polygon_points_y)))
    polygon_points_up = [(x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
    polygon_points_down = [(x, -y) for x, y in zip(polygon_points_x, polygon_points_y)]
    polygon_points = np.array(polygon_points_up[::-1] + polygon_points_down)
    return polygon_points

# The geometry will pass on the bounds and initial parameters to the optimizer
bounds = [(0.2e-6, 1e-6)]*10
initial_params=np.array([0.24481602, 0.26607235, 0.26211813, 0.43546523, 0.65633663, 0.65340398, 0.68939602, 0.62139496, 0.66486771, 0.58121573])*1e-6

geometry = function_defined_Polygon(func=taper_splitter, initial_params=initial_params,
                                    eps_out=1.44 ** 2, eps_in='Si (Silicon) - Palik', bounds=bounds,
                                    depth=220e-9,
                                    edge_precision=5)
# We must define the permittivities of the material making the optimizable
# geometry and of that surrounding it. Since this is a 2D simulation, the depth has no importance.
# edge_precision defines the discretization of the edges forming the optimizable polygon. It should be set such
# that there are at least a few points per mesh cell. An effective index of 2.8 is user to simulate a 2D slab of
# 220 nm thick

######## DEFINE FIGURE OF MERIT ########

fom = ModeMatch(modeorder=1,monitor_name='fom',wavelength=1550e-9)
# The base simulation script defines a field monitor named 'fom' at the point where we want to
# modematch to the 3rd order mode (fundamental TE mode)

######## DEFINE OPTIMIZATION ALGORITHM ########
optimizer = ScipyOptimizers(max_iter=20,method='L-BFGS-B',scaling_factor=1e6)
# This will run Scipy's implementation of the L-BFGS-B algoithm for at least 40 iterations. Since the variables are on the
# order of 1e-6, we scale them up to be on the order of 1

######## PUT EVERYTHING TOGETHER ########
opt = Optimization(base_script=script, fom=fom, geometry=geometry, optimizer=optimizer)

######## RUN THE OPTIMIZER ########
opt.run()

