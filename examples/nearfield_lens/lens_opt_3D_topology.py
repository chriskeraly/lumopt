######## IMPORTS ########
# General purpose imports
import numpy as np
import os
import sys
import scipy as sp

# Optimization specific imports
from lumopt import CONFIG
from lumopt.geometries.topology import TopologyOptimization2D, TopologyOptimization3DLayered
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.figures_of_merit.fieldintensities import FieldIntensity
from lumopt.optimization import Optimization
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.optimizers.fixed_step_gradient_descent import FixedStepGradientDescent
from lumopt.utilities.wavelengths import Wavelengths

######## DEFINE BASE SIMULATION ########
def runSim(params, eps_bg, eps_lens, x_pos, y_pos, z_pos, size_x, filter_R):

    ######## DEFINE A 3D LATOPOLOGY OPTIMIZATION REGION ########
    geometry = TopologyOptimization3DLayered(params=params, eps_min=eps_bg, eps_max=eps_lens, x=x_pos, y=y_pos, z=z_pos, filter_R=filter_R)
    wavelengths = Wavelengths(start = 1550e-9, stop = 1550e-9, points = 1)

    ######## DEFINE FIGURE OF MERIT ########
    # The base simulation script defines a field monitor named 'fom' at the point where we want to optimize the E-field intensity
    fom = FieldIntensity(monitor_name = 'fom', wavelengths=wavelengths)

    ######## DEFINE OPTIMIZATION ALGORITHM ########
    #optimizer = ScipyOptimizers(max_iter=50, method='SLSQP', scaling_factor=1, pgtol=1e-6, ftol=1e-4, target_fom=1, scale_initial_gradient_to=0.25)

    optimizer = FixedStepGradientDescent(max_dx=0.1, # edges of rectangles both move one mesh point
                                    max_iter = 60,
                                    all_params_equal=False,
                                    noise_magnitude=0,
                                    scaling_factor = 1)

    ######## LOAD TEMPLATE SCRIPT AND SUBSTITUTE PARAMETERS ########
    script = load_from_lsf('lens_base_3D_topology.lsf')

    opt = Optimization(base_script=script, wavelengths = wavelengths, fom=fom, geometry=geometry, optimizer=optimizer, use_deps=False, hide_fdtd_cad=False, plot_history=False, store_all_simulations=False)

    ######## RUN THE OPTIMIZER ########
    opt.run()

if __name__ == '__main__':
    size_x = 500
    size_y = 500
    size_z = 200
    
    filter_R = 200e-9
    
    eps_lens = 3.48**2
    eps_bg = 1**2

    if len(sys.argv) > 2 :
        size_x = int(sys.argv[1])
        filter_R = int(sys.argv[2])*1e-9
        print(size_x,filter_R)

    x_points=int(size_x/20)+1
    y_points=int(size_y/20)+1
    z_points=int(size_z/20)+1

    x_pos = np.linspace(0,size_x*1e-9,x_points)
    y_pos = np.linspace(0,size_y*1e-9,y_points)
    z_pos = np.linspace(-size_z/2*1e-9,size_z/2*1e-9,z_points)

    # ## We can either start from the result of a 2d simulation
    # filename2d = os.path.join(CONFIG['root'], 'examples/Ysplitter/opts_0/parameters_523.npz')
    # geom2d = TopologyOptimization2D.from_file( filename2d, filter_R=filter_R)
    # params2d = geom2d.last_params

    ## And then also a few systematic tests
    paramList=[np.ones((x_points,y_points)),       #< Start with the domain filled with eps_wg
               0.5*np.ones((x_points,y_points)),   #< Start with the domain filled with (eps_wg+eps_bg)/2
               np.zeros((x_points,y_points)),      #< Start with the domain filled with eps_bg
               ]                               

    init_params=paramList[1]

    runSim(init_params, eps_bg, eps_lens, x_pos, y_pos, z_pos, size_x*1e-9, filter_R)
