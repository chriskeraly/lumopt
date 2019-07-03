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

def runGratingOptimization(bandwidth_in_nm, etch_depth, n_grates, params):

    bounds = [(0.1, 1)]*4
    bounds[0] = (-3,3)      #< Starting position 
    bounds[1] = (0,0.1)     #< Scaling parameter R
    bounds[2] = (1.5,3)     #< Parameter a
    bounds[3] = (0,2)       #< Parameter b

    def grating_params_pos(params, output_waveguide_length = 0.5e-6, height = 220e-9, y0 = 0):
        x_begin = -3e-6
        y3 = y0+height
        y1 = y3-etch_depth

        x_start = params[0]*1e-6  #< First parameter is the starting position
        x0 = x_start
        R  = params[1]*1e6        #< second parameter (unit is 1/um)
        a  = params[2]            #< Third parameter (dim-less)
        b  = params[3]            #< Fourth parameter (dim-less)
  
        verts = np.array( [ [x_begin,y0],[x_begin,y3],[x0,y3],[x0,y1] ] )
        
        lambda_c = 1.55e-6
        F0 = 0.95

        ## Iterate over all but the last
        for i in range(n_grates-1):
            F = F0-R*(x0-x_start)
            Lambda = lambda_c / (a+F*b)
            x1 = x0 + (1-F)*Lambda    #< Width of the etched region
            x2 = x0 + Lambda          #< Rest of cell
            verts = np.concatenate((verts,[[x1,y1],[x1,y3],[x2,y3],[x2,y1]]),axis=0)
            x0 = x2

        F = F0-R*(x0-x_start)
        Lambda = lambda_c / (a+F*b)
        x1 = x0 + (1-F)*Lambda        #< Width of the etched region
        x_end   = x1+output_waveguide_length
        verts = np.concatenate((verts,[[x1,y1],[x1,y3],[x_end,y3],[x_end,y0]]),axis=0) 

        return verts

    geometry = FunctionDefinedPolygon(func = grating_params_pos, initial_params = params, bounds = bounds, z = 0.0, depth = 110e-9, eps_out = 1.44 ** 2, eps_in = 3.47668 ** 2, edge_precision = 5, dx = 1e-3)

    ######## DEFINE FIGURE OF MERIT ########
    fom = ModeMatch(monitor_name = 'fom', mode_number = 1, direction = 'Backward', target_T_fwd = lambda wl: np.ones(wl.size), norm_p = 1)

    ######## DEFINE OPTIMIZATION ALGORITHM ########
    optimizer = ScipyOptimizers(max_iter = 25, method = 'L-BFGS-B', scaling_factor = 1, pgtol = 1e-6)

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
    bandwidth_in_nm = 0 #< Only optimiza for center frequency of 1550nm
    initial_params = [0, 0.03, 2.4, 0.5369]
 
    runGratingOptimization(bandwidth_in_nm=bandwidth_in_nm, etch_depth=80e-9, n_grates = 25, params=initial_params)
