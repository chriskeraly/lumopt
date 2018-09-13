######## IMPORTS ########
import numpy as np

from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimization import Optimization
from lumopt.optimizers.generic_optimizers import ScipyOptimizers, FixedStepGradientDescent
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
import os
from lumopt.geometries.polygon import function_defined_Polygon, cross
from lumopt.utilities.materials import Material
from lumopt import CONFIG
import scipy

######## DEFINE BASE SIMULATION ########

script = load_from_lsf(os.path.join(CONFIG['root'], 'examples/crossing/crossing_base_TE_modematch_2D.lsf'))

######## DEFINE OPTIMIZABLE GEOMETRY ########
bounds = [(0.2e-6, 1e-6)]*10
geometry = function_defined_Polygon(func=cross, initial_params=np.linspace(0.25e-6, 0.6e-6, 10),
                                    eps_out=Material(1.44 ** 2), eps_in=Material(2.8 ** 2, 2), bounds=bounds,
                                    depth=220e-9,
                                    edge_precision=5)

######## DEFINE FIGURE OF MERIT ########

fom = ModeMatch(modeorder=2,precision=100)

######## DEFINE OPTIMIZATION ALGORITHM ########
optimizer = ScipyOptimizers(max_iter=40)

#optimizer = FixedStepGradientDescent(max_dx=5e-9, max_iter=100)




geometry = function_defined_Polygon(func=cross, initial_params=np.linspace(0.25e-6, 0.6e-6, 10),
                                    eps_out=Material(1.44 ** 2), eps_in=Material(2.8 ** 2, 2), bounds=bounds,
                                    depth=220e-9,
                                    edge_precision=5)

######## PUT EVERYTHING TOGETHER ########
opt = Optimization(base_script=script, fom=fom, geometry=geometry, optimizer=optimizer)

######## RUN THE OPTIMIZER ########
opt.run()
