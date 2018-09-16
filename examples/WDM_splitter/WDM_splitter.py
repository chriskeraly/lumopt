######## IMPORTS ########
# General purpose imports
import numpy as np
from lumopt.optimization import Super_Optimization, Optimization
from lumopt.geometries.polygon import function_defined_Polygon
from lumopt.optimizers.generic_optimizers import ScipyOptimizers

from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
import os
from lumopt import CONFIG
import scipy

######## DEFINE BASE SIMULATION ########

#Here I just use the same script for both simulations, but it's just to keep the example simple. You could use two
script_1550=load_from_lsf(os.path.join(CONFIG['root'],'examples/WDM_splitter/WDM_splitter_base_TE_1550.lsf'))
script_1310=load_from_lsf(os.path.join(CONFIG['root'],'examples/WDM_splitter/WDM_splitter_base_TE_1550.lsf')).replace('1550e-9','1310e-9')


######## DEFINE OPTIMIZABLE GEOMETRY ########
separation=200e-9
size_x=10e-6

def lower_coupler_arm(params,n_points=10):
    points_x=np.concatenate(([0.5e-6],np.linspace(0.55e-6,size_x-0.55e-6,20),[size_x-0.5e-6]))
    points_y=np.concatenate(([-separation/2],params-separation/2,params[::-1]-separation/2,[-separation/2]))
    n_interpolation_points=100
    polygon_points_x = np.linspace(min(points_x), max(points_x), n_interpolation_points)
    interpolator = scipy.interpolate.interp1d(points_x, points_y, kind='cubic')
    polygon_points_y = [max(min(point,0e-6),-0.5e-6) for point in interpolator(polygon_points_x)]

    polygon_points_up = [(x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
    polygon_points_down = [(0.5e-6,-separation/2-0.5e-6),(size_x-0.5e-6,-separation/2-0.5e-6)]
    polygon_points = np.array(polygon_points_up[::-1] + polygon_points_down)
    return polygon_points#[::-1]

def upper_coupler_arm(params,n_points=10):
    points_x=np.concatenate(([0.5e-6],np.linspace(0.55e-6,size_x-0.55e-6,20),[size_x-0.5e-6]))
    points_y=np.concatenate(([separation/2],-params+separation/2,-params[::-1]+separation/2,[separation/2]))
    n_interpolation_points=100
    polygon_points_x = np.linspace(min(points_x), max(points_x), n_interpolation_points)
    interpolator = scipy.interpolate.interp1d(points_x, points_y, kind='cubic')
    polygon_points_y = [max(min(point,0.5e-6),-0e-6) for point in interpolator(polygon_points_x)]

    polygon_points_up = [(x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
    polygon_points_down = [(0.5e-6,separation/2+0.5e-6),(size_x-0.5e-6,separation/2+0.5e-6)]
    polygon_points = np.array(polygon_points_up + polygon_points_down[::-1])
    return polygon_points

bounds = [(-0.25e-6, 0.25e-6)]*10

#final value from splitter_opt_2D.py optimization
initial_params=np.array(10*[0e-6])
#initial_params=np.linspace(-0.25e-6,0.25e-6,10)
geometry_1550_lower =  function_defined_Polygon(func=lower_coupler_arm,initial_params=initial_params,eps_out=1.44 ** 2, eps_in=2.8 ** 2,bounds=bounds,depth=220e-9,edge_precision=5)
geometry_1550_upper =  function_defined_Polygon(func=upper_coupler_arm,initial_params=initial_params,eps_out=1.44 ** 2, eps_in=2.8 ** 2,bounds=bounds,depth=220e-9,edge_precision=5)
geometry_1550=geometry_1550_lower*geometry_1550_upper
geometry_1310_lower =  function_defined_Polygon(func=lower_coupler_arm,initial_params=initial_params,eps_out=1.44 ** 2, eps_in=2.8 ** 2,bounds=bounds,depth=220e-9,edge_precision=5)
geometry_1310_upper =  function_defined_Polygon(func=upper_coupler_arm,initial_params=initial_params,eps_out=1.44 ** 2, eps_in=2.8 ** 2,bounds=bounds,depth=220e-9,edge_precision=5)
geometry_1310=geometry_1310_lower*geometry_1310_upper

######## DEFINE FIGURE OF MERIT ########
# Although we are optimizing for the same thing, two separate fom objects must be create

fom_1550=ModeMatch(modeorder=2,wavelength=1550e-9,monitor_name='fom_1550')
fom_1310=ModeMatch(modeorder=2,wavelength=1310e-9,monitor_name='fom_1310')

######## DEFINE OPTIMIZATION ALGORITHM ########
#For the optimizer, they should all be set the same, but different objects. Eventually this will be improved
optimizer_1550=ScipyOptimizers(max_iter=40)
optimizer_1310=ScipyOptimizers(max_iter=40)

######## PUT EVERYTHING TOGETHER ########
opt_1550=Optimization(base_script=script_1550,fom=fom_1550,geometry=geometry_1550,optimizer=optimizer_1550)
opt_1310=Optimization(base_script=script_1310,fom=fom_1310,geometry=geometry_1310,optimizer=optimizer_1310)

opt=opt_1310+opt_1550

######## RUN THE OPTIMIZER ########
opt.run()
# opt_1550.initialize()
# sim=opt_1550.make_sim()
# from time import sleep
# sleep(1000)
# print 'ha'