"""
    Copyright (c) 2019 Lumerical Inc. """

######## IMPORTS ########
# General purpose imports
import os
import numpy as np
import scipy as sp

# Optimization specific imports
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.utilities.wavelengths import Wavelengths
from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.optimization import Optimization
from lumopt.utilities.materials import Material

from numpy.random import rand

def runGratingOptimization(bandwidth_in_nm, etch_depth, n_grates, initial_params = None):
    ### Yet another parametrization which allows to enforce minimum feature size when the optimizer only supports box constraints  
    ### params = [x0, a1, b1, ..., aN]
    if initial_params is None:
        params = np.zeros(2*n_grates)

        for i in range(n_grates):
            params[i*2]   = 0.2     #< Width up
            params[i*2+1] = 0.4*((i+1)/(n_grates+1))     #< Width of the deep etch

        params[0] = 0      #< Overwrite the first since it has a special meaning: Start of the grating at x=0
    else:
        params = initial_params

    bounds = [(0, 1)]*(2*n_grates)  
    bounds[0] = (-3,3)      #< Bounds for the stating position

    def grating_params_pos(params, output_waveguide_length = 0.5e-6, height = 220e-9, y0 = 0):
        x_begin = -3e-6
        y3 = y0+height
        y1 = y3-etch_depth

        x0 = params[0]*1e-6     #< First parameter is the starting position
        verts = np.array( [ [x_begin,y0],[x_begin,y3],[x0,y3],[x0,y1] ] )
        
        ## Iterate over all but the last
        for i in range(n_grates-1):
            x1 = x0 + params[i*2+1]*1e-6    #< Width of the  etch
            x2 = x1 + params[i*2+2]*1e-6    #< Width up
            verts = np.concatenate((verts,[[x1,y1],[x1,y3],[x2,y3],[x2,y1]]),axis=0)
            x0 = x2

        x1 = x0 + params[(n_grates-1)*2+1]*1e-6    #< Width of the  etch
        x_end   = x1+output_waveguide_length
        verts = np.concatenate((verts,[[x1,y1],[x1,y3],[x_end,y3],[x_end,y0]]),axis=0) 

        return verts

    geometry = FunctionDefinedPolygon(func = grating_params_pos, initial_params = params, bounds = bounds, z = 0.0, depth = 220e-9, eps_out = 1.44 ** 2, eps_in = 3.47668 ** 2, edge_precision = 5, dx = 1e-3)

    ######## DEFINE FIGURE OF MERIT ########
    fom = ModeMatch(monitor_name = 'fom', mode_number = 1, direction = 'Backward', target_T_fwd = lambda wl: np.ones(wl.size), norm_p = 1)

    ######## DEFINE OPTIMIZATION ALGORITHM ########
    optimizer = ScipyOptimizers(max_iter = 250, method = 'L-BFGS-B', scaling_factor = 1, pgtol = 1e-6) #SLSQP

    ######## DEFINE BASE SIMULATION ########
    base_script = load_from_lsf(os.path.join(os.path.dirname(__file__), 'grating_coupler_2D_2etch.lsf'))

    ######## PUT EVERYTHING TOGETHER ########
    lambda_start = 1550 - bandwidth_in_nm/2
    lambda_end   = 1550 + bandwidth_in_nm/2
    lambda_pts   = int(bandwidth_in_nm/10)+1
    wavelengths = Wavelengths(start = lambda_start*1e-9, stop = lambda_end*1e-9, points = lambda_pts)
    opt = Optimization(base_script = base_script, wavelengths = wavelengths, fom = fom, geometry = geometry, optimizer = optimizer, hide_fdtd_cad = True, use_deps = True)

    ######## RUN THE OPTIMIZER ########
    opt.run()

if __name__ == "__main__":
    bandwidth_in_nm = 0
    initial_params=None

    # An initial guess for etch_depth of 80nm (around 58%) obtained via optimization of an apodized grating
    # 
    initial_params = [0, 0.0284433,  0.540422,  0.0413995 , 0.529737,  0.0545119,  0.518923,  0.0677836,  0.507977 ,
                        0.081218,  0.496898,  0.0948184,  0.485681,  0.108588,  0.474325,  0.122531,  0.462826,  0.13665, 0.451182 ,
                        0.15095,  0.439388,  0.165434,  0.427443,  0.180106,  0.415343,  0.19497,  0.403084,  0.210031,  0.390663, 
                        0.225293,  0.378076,  0.240761,  0.36532,  0.256438,  0.35239,  0.27233,  0.339284,  0.288442,  0.325996, 
                        0.304779,  0.312522,  0.321347,  0.298859,  0.338149,  0.285001,  0.355194,  0.270945,  0.372485,  0.256684,  0.390029 ]

    runGratingOptimization(bandwidth_in_nm=bandwidth_in_nm, etch_depth=80e-9, n_grates = 25, initial_params=initial_params)
