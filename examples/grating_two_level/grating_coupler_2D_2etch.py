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
            params[i*4+1] = 0.4*(i/n_grates)     #< Width of the deep etch
            params[i*4+2] = 0.1    #< Width up
            params[i*4+3] = 0.4*(i/n_grates)     #< Width of the shallow etch

        params[0] = -1      #< Overwrite the first since it has a special meaning: Start of the grating at -3um
    else:
        params = initial_params

    bounds = [(0.1, 1)]*(4*n_grates)  
    bounds[0] = (-3,3)

    def grating_params_pos(params, output_waveguide_length = 0.5e-6, height = 220e-9, y0 = 0):
        x_begin = -3e-6
        y3 = y0+height
        y2 = y3-etch_depth_shallow
        y1 = y3-etch_depth_deep

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


    # initial_params = [1.43386093, 0.04829219, 0.15869363, 0.08237436, 0.34699639, 0.02761985,
    #                   0.1355977 , 0.12122159, 0.33271724, 0.05124579, 0.08743708, 0.16443544,
    #                   0.35104438, 0.07280426, 0.03867738, 0.1674615 , 0.3001356 , 0.08501549,
    #                   0.09360974, 0.16944294, 0.30078774, 0.09478836, 0.05809306, 0.26614588,
    #                   0.26743501, 0.09817684, 0.03045605, 0.30858301, 0.22179623, 0.09401361,
    #                   0.05797512, 0.31167473, 0.22165514, 0.10494211, 0.052716  , 0.33203842,
    #                   0.20099363, 0.11816804, 0.05762081, 0.32589348, 0.20409799, 0.10641423,
    #                   0.0437345 , 0.3466378 , 0.20843631, 0.1119758 , 0.02600666, 0.37624217,
    #                   0.16633128, 0.13656591, 0.05491645, 0.38896486, 0.14084058, 0.15764873,
    #                   0.05675199, 0.37931832, 0.16922617, 0.15846364, 0.08106278, 0.30471894,
    #                   0.16028493, 0.19916474, 0.05622415, 0.37437696, 0.14781818, 0.18698968,
    #                   0.09233051, 0.3199459 , 0.17310785, 0.20024507, 0.09108384, 0.30481818,
    #                   0.15805038, 0.19946823, 0.10459085, 0.30474619, 0.14474484, 0.2130202,
    #                   0.05255327, 0.3859057 , 0.13252414, 0.18668785, 0.06654688, 0.37311683,
    #                   0.13316956, 0.21327723, 0.04003199, 0.4000668]

    ## Good result around 88%
    initial_params = [0.85121955, 0.06629826, 0.11961948, 0.14328734, 0.28587674, 0.06207677,
                      0.08767746, 0.21207104, 0.30621588, 0.05727117, 0.03186829, 0.25678915,
                      0.30039609, 0.06933111, 0.01721758, 0.26799844, 0.2758347 , 0.08497404,
                      0.04359087, 0.29066584, 0.23295373, 0.09848284, 0.04397317, 0.32574819,
                      0.22947156, 0.09496271, 0.02184496, 0.34815977, 0.22608223, 0.11066452,
                      0.0239981 , 0.36009914, 0.2043906 , 0.11910874, 0.03854418, 0.35834537,
                      0.19337371, 0.1331621 , 0.04227388, 0.36480883, 0.17757111, 0.15638102,
                      0.04323958, 0.36173586, 0.17596761, 0.17542101, 0.03963165, 0.37407011,
                      0.1669319 , 0.17616864, 0.04445841, 0.37593297, 0.15623866, 0.17193152,
                      0.05952692, 0.36289065, 0.15782357, 0.17724691, 0.07456616, 0.31772346,
                      0.16638646, 0.18757693, 0.06865937, 0.36361426, 0.14209618, 0.19159085,
                      0.08452124, 0.3156748 , 0.16434585, 0.19939001, 0.09322246, 0.30792149,
                      0.16082538, 0.1973796 , 0.10437939, 0.30621533, 0.14628429, 0.21225625,
                      0.05616597, 0.3869066 , 0.13461606, 0.18789383, 0.0665094,  0.37287196,
                      0.1341891 , 0.21479808, 0.04318507, 0.40273956]

    ## Uses a shallow-etched grating and inserts very shallow deep-etch trenches as initial condition
    # initial_params = [-0.02614452  0.05149638  0.53157201  0.055653    0.54536365  0.05354342
    #                    0.53982055  0.06101422  0.52618164  0.07250187  0.51095521  0.09012186
    #                    0.49897871  0.10672081  0.48604458  0.11078737  0.48717406  0.12747969
    #                    0.48490738  0.13336516  0.47647919  0.14450685  0.45710196  0.16006559
    #                    0.44897485  0.17166968  0.43510104  0.18515862  0.42741942  0.19681885
    #                    0.41868277  0.20688768  0.4075006   0.21855605  0.39874415  0.22626997
    #                    0.38691299  0.23823558  0.37462352  0.24958005  0.36115356  0.26292962
    #                    0.34920035  0.28013319  0.3414847   0.29843411  0.33349672  0.31724115
    #                    0.32501007  0.33399776]

    #initial_params = rand(1,88)
    #initial_params *= 0.4


    runGratingOptimization(bandwidth_in_nm=bandwidth_in_nm, etch_depth_shallow=80e-9, etch_depth_deep=220e-9, n_grates = 22, initial_params=initial_params)
