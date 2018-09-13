
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
from lumopt.optimizers.generic_optimizers import ScipyOptimizers,FixedStepGradientDescent
from lumopt.utilities.materials import Material

######## DEFINE BASE SIMULATION ########

script = load_from_lsf(os.path.join(CONFIG['root'], 'examples/grating/grating_base.lsf'))

######## DEFINE OPTIMIZABLE GEOMETRY ########

# The class function_defined_Polygon needs a parameterized Polygon (with points ordered
# in a counter-clockwise direction. This funtion will automatically generate all the polygons needed

n_grates=22

def grate_function_generator(grate_number,grate_height=70e-9,grate_center_y=220e-9-35e-9):
    def grate_function(params):
        grate_position=params[grate_number]
        grate_width=params[grate_number+len(params)/2]
        left=grate_position-grate_width/2
        right=grate_position+grate_width/2
        top=grate_center_y+grate_height/2
        bottom=grate_center_y-grate_height/2
        polygon_points_x=[left,left,right,right]
        polygon_points_y=[top,bottom,bottom,top]
        polygon_points=[(x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
        return np.array(polygon_points)
    return grate_function

initial_positions=550e-9*np.arange(n_grates)+300e-9
initial_widths=100e-9*np.ones(n_grates)
edge_precision=20

# The geometry will pass on the bounds and initial parameters to the optimizer
bounds = [(250e-9, 15e-6)]*n_grates+[(100e-9,500e-9)]*n_grates
initial_params=np.concatenate((initial_positions,initial_widths))


oxide=Material(material=1.44**2,mesh_order=1)
silicon=Material(material=3.4**2) #default material is Silicon
first_tooth=function_defined_Polygon(func=grate_function_generator(0), initial_params=initial_params,
                                    eps_out=silicon, eps_in=oxide, bounds=bounds,
                                    depth=1e-6,
                                    edge_precision=edge_precision)
full_geometry=first_tooth

for i in range(1,n_grates):
    new_tooth=function_defined_Polygon(func=grate_function_generator(i), initial_params=initial_params,
                                    eps_out=silicon, eps_in=oxide, bounds=bounds,
                                    depth=1e-6,
                                    edge_precision=edge_precision)
    full_geometry=full_geometry*new_tooth



# We must define the permittivities of the material making the optimizable
# geometry and of that surrounding it. Since this is a 2D simulation, the depth has no importance.
# edge_precision defines the discretization of the edges forming the optimizable polygon. It should be set such
# that there are at least a few points per mesh cell. An effective index of 2.8 is user to simulate a 2D slab of
# 220 nm thick

######## DEFINE FIGURE OF MERIT ########

fom = ModeMatch(modeorder=1,monitor_name='fom',wavelength=1550e-9)#,direction='Backward')
# The base simulation script defines a field monitor named 'fom' at the point where we want to
# modematch to the 3rd order mode (fundamental TE mode)

######## DEFINE OPTIMIZATION ALGORITHM ########
optimizer = ScipyOptimizers(max_iter=30,method='L-BFGS-B',scaling_factor=1e6)
#optimizer = FixedStepGradientDescent(max_iter=40,max_dx=20e-9)#ScipyOptimizers(max_iter=30,method='L-BFGS-B',scaling_factor=1e6)
# This will run Scipy's implementation of the L-BFGS-B algoithm for at least 40 iterations. Since the variables are on the
# order of 1e-6, we scale them up to be on the order of 1

######## PUT EVERYTHING TOGETHER ########
opt = Optimization(base_script=script, fom=fom, geometry=full_geometry, optimizer=optimizer)

######## RUN THE OPTIMIZER ########
opt.run()

