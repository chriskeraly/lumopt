
######## IMPORTS ########
# General purpose imports
import numpy as np
import os
import scipy
from lumopt import CONFIG

# Optimization specific imports
from lumopt.geometries.continuous_epsilon import ContinousEpsilon2D
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.geometries.polygon import function_defined_Polygon
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimization import Optimization
from lumopt.optimizers.generic_optimizers import ScipyOptimizers,FixedStepGradientDescent,Adaptive_Gradient_Descent
from lumopt.utilities.materials import Material

######## DEFINE BASE SIMULATION ########

script = load_from_lsf(os.path.join(CONFIG['root'], 'examples/grating/grating_base_continuous.lsf'))

######## DEFINE OPTIMIZABLE GEOMETRY ########
x_points=18000/20
y_points=220/20

eps=np.ones((x_points,y_points))*3.4**2
geometry = ContinousEpsilon2D(eps=eps,x=np.linspace(0,18e-6,x_points),y=np.linspace(0,0.22e-6,y_points))
######## DEFINE FIGURE OF MERIT ########

fom = ModeMatch(modeorder=1,monitor_name='fom',wavelength=1550e-9)#,direction='Backward')


######## DEFINE OPTIMIZATION ALGORITHM ########
optimizer = ScipyOptimizers(max_iter=200,method='L-BFGS-B',scaling_factor=1e-6,pgtol=1e-10)


######## PUT EVERYTHING TOGETHER ########
opt = Optimization(base_script=script, fom=fom, geometry=geometry, optimizer=optimizer)

######## RUN THE OPTIMIZER ########
opt.run()

