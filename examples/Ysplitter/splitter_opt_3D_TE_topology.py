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
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimization import Optimization
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.utilities.wavelengths import Wavelengths

######## DEFINE BASE SIMULATION ########
def runSim(params, eps_bg, eps_wg, x_pos, y_pos, z_pos, size_x, filter_R):

    ######## DEFINE A 3D LATOPOLOGY OPTIMIZATION REGION ########
    geometry = TopologyOptimization3DLayered(params=params, eps_min=eps_bg, eps_max=eps_wg, x=x_pos, y=y_pos, z=z_pos, filter_R=filter_R)

    ######## DEFINE FIGURE OF MERIT ########
    # The base simulation script defines a field monitor named 'fom' at the point where we want to modematch to the fundamental TE mode
    fom = ModeMatch(monitor_name = 'fom', mode_number = 'Fundamental TE mode', direction = 'Forward', norm_p = 2)

    ######## DEFINE OPTIMIZATION ALGORITHM ########
    optimizer = ScipyOptimizers(max_iter=50, method='L-BFGS-B', scaling_factor=1, pgtol=1e-6, ftol=1e-4, target_fom=0.5, scale_initial_gradient_to=0.25)

    ######## LOAD TEMPLATE SCRIPT AND SUBSTITUTE PARAMETERS ########
    script = load_from_lsf(os.path.join(CONFIG['root'], 'examples/Ysplitter/splitter_base_3D_TE_topology.lsf'))
    script = script.replace('opt_size_x=3.0e-6','opt_size_x={:1.6g}'.format(size_x))

    wavelengths = Wavelengths(start = 1450e-9, stop = 1650e-9, points = 11)
    opt = Optimization(base_script=script, wavelengths = wavelengths, fom=fom, geometry=geometry, optimizer=optimizer, use_deps=False, hide_fdtd_cad=True, plot_history=False, store_all_simulations=False)

    ######## RUN THE OPTIMIZER ########
    opt.run()

if __name__ == '__main__':
    size_x = 3000
    size_y = 1800
    size_z = 220
    
    filter_R = 400e-9
    
    eps_wg = 3.48**2
    eps_bg = 1.44**2

    if len(sys.argv) > 2 :
        size_x = int(sys.argv[1])
        filter_R = int(sys.argv[2])*1e-9
        print(size_x,filter_R)

    x_points=int(size_x/20)+1
    y_points=int(size_y/20)+1
    z_points=int(size_z/20)+1

    x_pos = np.linspace(-size_x/2*1e-9,size_x/2*1e-9,x_points)
    y_pos = np.linspace(0,size_y*1e-9,y_points)
    z_pos = np.linspace(-size_z/2*1e-9,size_z/2*1e-9,z_points)

    # ## We can either start from the result of a 2d simulation
    # filename2d = os.path.join(CONFIG['root'], 'examples/Ysplitter/opts_0/parameters_523.npz')
    # geom2d = TopologyOptimization2D.from_file( filename2d, filter_R=filter_R)
    # params2d = geom2d.last_params

    ## And then also a few systematic tests
    paramList=[None,                               #< Use the structure defined in the project file as initial condition
               np.ones((x_points,y_points)),       #< Start with the domain filled with eps_wg
               0.5*np.ones((x_points,y_points)),   #< Start with the domain filled with (eps_wg+eps_bg)/2
               np.zeros((x_points,y_points)),      #< Start with the domain filled with eps_bg
               ]                               

    for curParams in paramList:
        runSim(curParams, eps_bg, eps_wg, x_pos, y_pos, z_pos, size_x*1e-9, filter_R)
