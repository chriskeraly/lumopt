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

def runGratingOptimization(bandwidth_in_nm, etch_depth_shallow, etch_depth_deep, n_grates, initial_params = None):
    ### Yet another parametrization which allows to enforce minimum feature size when the optimizer only supports box constraints  
    ### params = [x0, a1, b1, ..., aN]
    if initial_params is None:
        params = np.zeros(4*n_grates)

        for i in range(n_grates):
            params[i*4]   = 0.2     #< Width up
            params[i*4+1] = 0.4*(i/n_grates)     #< Width of the shallow etch
            params[i*4+2] = 0.1    #< Width up
            params[i*4+3] = 0.4*(i/n_grates)     #< Width of the deep etch

        params[0] = 0      #< Overwrite the first since it has a special meaning: Start of the grating at 0um
    else:
        params = initial_params

    bounds = [(0, 1)]*(4*n_grates)  
    bounds[0] = (-3,3)

    def grating_params_pos(params, output_waveguide_length = 0.5e-6, height = 220e-9, y0 = 0):
        x_begin = -3e-6

        y3 = y0+height
        y2 = y3-etch_depth_deep
        y1 = y3-etch_depth_shallow

        x0 = params[0]*1e-6     #< First parameter is the starting position
        verts = np.array( [ [x_begin,y0],[x_begin,y3],[x0,y3],[x0,y1] ] )
        
        ## Iterate over all but the last
        for i in range(n_grates-1):
            x1 = x0 + params[i*4+1]*1e-6    #< Width of the deep etch
            x2 = x1 + params[i*4+2]*1e-6    #< Width up
            x3 = x2 + params[i*4+3]*1e-6    #< Width of the shallow etch
            x4 = x3 + params[i*4+4]*1e-6    #< Width up
            verts = np.concatenate((verts,[[x1,y1],[x1,y3],[x2,y3],[x2,y2],[x3,y2],[x3,y3],[x4,y3],[x4,y1]]),axis=0)
            x0 = x4

        x1 = x0 + params[(n_grates-1)*4+1]*1e-6    #< Width of the deep etch
        x2 = x1 + params[(n_grates-1)*4+2]*1e-6    #< Width up
        x3 = x2 + params[(n_grates-1)*4+3]*1e-6    #< Width of the shallow etch
        x_end   = x3+output_waveguide_length
        verts = np.concatenate((verts,[[x1,y1],[x1,y3],[x2,y3],[x2,y2],[x3,y2],[x3,y3],[x_end,y3],[x_end,y0]]),axis=0) 

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

    # ## Uses a shallow-etched grating and inserts very shallow deep-etch trenches as initial condition
    # initial_params = [-0.02614452,  0.05149638,  0.53157201,  0.055653  ,  0.54536365,  0.05354342,
    #                    0.53982055,  0.06101422,  0.52618164,  0.07250187,  0.51095521,  0.09012186,
    #                    0.49897871,  0.10672081,  0.48604458,  0.11078737,  0.48717406,  0.12747969,
    #                    0.48490738,  0.13336516,  0.47647919,  0.14450685,  0.45710196,  0.16006559,
    #                    0.44897485,  0.17166968,  0.43510104,  0.18515862,  0.42741942,  0.19681885,
    #                    0.41868277,  0.20688768,  0.4075006 ,  0.21855605,  0.39874415,  0.22626997,
    #                    0.38691299,  0.23823558,  0.37462352,  0.24958005,  0.36115356,  0.26292962,
    #                    0.34920035,  0.28013319,  0.3414847 ,  0.29843411,  0.33349672,  0.31724115,
    #                    0.32501007,  0.33399776]
    initial_params = [ -0.0239948  , 0.06420394,  0.51077913,  0.07130651,  0.52258685,  0.06810984,
                        0.51575204,  0.08343425,  0.50207087,  0.10838259,  0.47610344,  0.13345371,
                        0.45337923,  0.15746284,  0.42984944,  0.16298374,  0.41697026,  0.19336803,
                        0.40854747,  0.20099055,  0.40329468,  0.20697513,  0.39361016,  0.21359092,
                        0.39192629,  0.21175156,  0.38271668,  0.22063051,  0.37988725,  0.21807104,
                        0.37207578,  0.22694823,  0.36307474,  0.24185864,  0.35741802,  0.25815608,
                        0.34402785,  0.2778519 ,  0.32960496,  0.28935076,  0.307496  ,  0.3071018,
                        0.29218344,  0.32577172,  0.28079774,  0.34911492,  0.27176665,  0.37340163,
                        0.26120352,  0.39409659]
    


    d = 0.05    #< Introduce 50nm trenches
    interleaved_params = initial_params[0:2]
    for idx in range(2,len(initial_params),2):
        wide = initial_params[idx]
        shallow = initial_params[idx+1]
        interleaved_params = np.concatenate((interleaved_params,[(wide-d)/2,d,(wide-d)/2,shallow]),axis=0)
    interleaved_params = np.concatenate((interleaved_params,[0.25,0.25]))
    
    initial_params = interleaved_params


    ## Initial conditions with efficieny around 82%. Leads to the currently best known structure (>91%). Values
    ## were obtained via a constrained (feature size 100nm) optimization of a single-etch structure.
    initial_params = [  0.09361918, 0.1,        0.27328058, 0.1,        0.13785672, 0.12336592,
                    0.37274167 ,0.10038749, 0.1       , 0.1       , 0.28285332, 0.1,
                    0.1        ,0.20575191, 0.1778981 , 0.1       , 0.1       , 0.22445837,
                    0.33808579 ,0.11603228, 0.1       , 0.10366125, 0.27984459, 0.1,
                    0.1        ,0.1875518 , 0.21804731, 0.10357594, 0.1       , 0.24304465,
                    0.27746333 ,0.1       , 0.1       , 0.1908697 , 0.27634044, 0.11374789,
                    0.1        ,0.21772678, 0.22564103, 0.11696634, 0.1       , 0.25826664,
                    0.23364527 ,0.14218281, 0.1       , 0.26132716, 0.21019975, 0.15264778,
                    0.1        ,0.26516554, 0.20402114, 0.15427596, 0.1       , 0.25941542,
                    0.20987306 ,0.15313865, 0.1       , 0.27228623, 0.21365611, 0.16284176,
                    0.1174785  ,0.24581137, 0.19230639, 0.11787995, 0.12760224, 0.24290709,
                    0.1880198  ,0.12818071, 0.16776542, 0.25400389, 0.18479421, 0.10535137,
                    0.15350407 ,0.2353964 , 0.15881185, 0.1       , 0.15865047, 0.25534048,
                    0.16246567 ,0.10075842, 0.15419022, 0.26211885, 0.15051856, 0.1002462,
                    0.14781438 ,0.27741503, 0.14379248, 0.1       , 0.14368657, 0.29688433,
                    0.14052862 ,0.10016114, 0.14034895, 0.31517367, 0.13592246, 0.10007413,
                    0.13457174 ,0.33105594, 0.34725264, 0.34897194]

    runGratingOptimization(bandwidth_in_nm=bandwidth_in_nm, etch_depth_shallow=80e-9, etch_depth_deep=220e-9, n_grates = 25, initial_params=initial_params)#=interleaved_params)
