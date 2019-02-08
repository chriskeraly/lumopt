""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import os
import time
import shutil
import inspect
import copy
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from lumopt.utilities.simulation import Simulation
from lumopt.utilities.gradients import Gradient_fields
from lumopt.utilities.plotter import Plotter
from lumopt.utilities.scipy_wrappers import trapz3D
from lumopt.lumerical_methods.lumerical_scripts import add_index_to_fields_monitors, enable_accurate_conformal_interface_detection

class Super_Optimization(object):
    ''' Optimization base class which allows the user to use the addition operator to co-optimize two figures of merit
        that take the same parameters. The figures of merit are simply added and the plotting functions use the first
        figure of merit.'''

    def __init__(self,optimizations):
        self.optimizations=optimizations

    def __add__(self,other):
        optimizations=[self,other]
        return Super_Optimization(optimizations)

    def initialize(self,start_params=None,bounds=None):

        print('Initializing super optimization')

        self.optimizer = copy.deepcopy(self.optimizations[0].optimizer)
        self.plotter=self.optimizations[0].plotter

        # First
        for optimization in self.optimizations:
            optimization.initialize()

        if start_params is None:
            start_params=self.optimizations[0].geometry.get_current_params()
        if bounds is None:
            bounds=np.array(self.optimizations[0].geometry.bounds)

        def callable_fom(params):
            fom=0
            for optimization in self.optimizations:
                fom+=optimization.callable_fom(params)
            return fom

        def callable_jac(params):
            jac=0
            for optimization in self.optimizations:
                jac+=np.array(optimization.callable_jac(params))
            return jac

        def plotting_function():
            self.plotter.update(self)

        if hasattr(self.optimizer,'initialize'):
            self.optimizer.initialize(start_params=start_params,callable_fom=callable_fom,callable_jac=callable_jac,bounds=bounds,plotting_function=plotting_function)


    def run(self):
        ''' Inititalizes and then runs the optimization. '''

        self.initialize()
        if self.plotter.movie:
            with self.plotter.writer.saving(self.plotter.fig, "optimization.mp4", 100):
                self.optimizer.run()
        else:
            self.optimizer.run()

        print('FINAL FOM = {}'.format(self.optimizer.fom_hist[-1]))
        print('FINAL PARAMETERS = {}'.format(self.optimizer.params_hist[-1]))
        return self.optimizer.fom_hist[-1],self.optimizer.params_hist[-1]


class Optimization(Super_Optimization):
    """ Acts as a master class for all the optimization pieces. Calling the function run will perform the full optimization.
        To perform an optimization, four key pieces are requred. These are: 
               1) a script to generate the base simulation,
               2) an object that defines and collects the figure of merit,
               3) an object that generates the shape under optimization for a given set of optimization parameters and
               4) a SciPy gradient based minimizer.

        :base_script: string with script to generate the base simulation.
        :fom:         figure of merit object (see class ModeMatch).
        :geometry:    optimizable geometry (see class FunctionDefinedPolygon).
        :optimizer:   SciyPy minimizer wrapper (see class ScipyOptimizers).
    """

    def __init__(self, base_script, fom, geometry, optimizer, use_deps = True):

        self.base_script = base_script
        self.fom = fom
        self.geometry = geometry
        self.optimizer = optimizer
        self.use_deps = use_deps
        if use_deps:
            print("Accurate interface detection enabled")

        self.plotter = Plotter()
        self.forward_fields = None
        self.adjoint_fields = None
        self.gradients = None
        self.fomHist = []
        self.paramsHist = []
        self.gradient_fields = None

        frame = inspect.stack()[1]
        calling_file_name = os.path.abspath(frame[0].f_code.co_filename)
        Optimization.goto_new_opts_folder(calling_file_name, base_script)
        self.workingDir = os.getcwd()

    def run(self):
        ''' Inititalizes and then runs the optimization. '''

        self.initialize()
        if self.plotter.movie:
            with self.plotter.writer.saving(self.plotter.fig, "optimization.mp4", 100):
                self.optimizer.run()

        else:
            self.optimizer.run()

        print('FINAL FOM = {}'.format(self.optimizer.fom_hist[-1]))
        print('FINAL PARAMETERS = {}'.format(self.optimizer.params_hist[-1]))
        return self.optimizer.fom_hist[-1],self.optimizer.params_hist[-1]


    def initialize(self):
        '''Before the optimization can be run, usually a few things needs to be initialized.

            - The FOM passes the wavelengths at which the optimization is done to the geomtry, which needs to know, in order to calculate the gradients
            - In some case the fom needs to extract some information from the base simulation before being fully defined (a mode for example)
            - The geometry passes the starting parameters to the optimizer'''

        # The Materials in the geometries have to be initilized at the wavelengths of interest
        wavelengths=self.fom.get_wavelengths()
        self.geometry.initialize(wavelengths,self)

        # For Modematch FOM the FOM must be initialized
        sim = self.make_sim()
        self.fom.initialize(sim)

        # The optimizer needs to know how to call the methods of this class to calculate the figures of merit

        start_params = self.geometry.get_current_params()
        callable_fom = self.callable_fom
        callable_jac = self.callable_jac
        bounds = np.array(self.geometry.bounds)

        def plotting_function():
            self.plotter.update(self)

        self.optimizer.initialize(start_params=start_params,callable_fom=callable_fom,callable_jac=callable_jac,bounds=bounds,plotting_function=plotting_function)


    def make_base_sim(self):
        ''' Creates the base simulation (without the optimizable geometry) using the provided script and saves the project
            file in the specified working directiory. '''

        return Simulation(workingDir = self.workingDir, script = self.base_script)

    def make_sim(self, geometry = None):
        '''Creates the forward simulation by adding the geometry to the base simulation and adding D monitors to any field monitor in the simulation.
        If the FOM object needs it's own monitors (or needs to manipulate those already present) they will also be added/manipulated.

        :param geometry: By default the current gometry of the Optimization #will be put in, but this can be overriden by inputing another geometry here

        :returns: sim, Handle to the simulation '''

        # create the simulation object
        sim = self.make_base_sim()
        add_index_to_fields_monitors(sim.fdtd, 'opt_fields')
        sim.fdtd.setnamed('opt_fields', 'spatial interpolation', 'None')
        if(self.use_deps):
            enable_accurate_conformal_interface_detection(sim.fdtd)

        time.sleep(0.1)

        # add the optimizable geometry
        if geometry is None:
            self.geometry.add_geo(sim)
        else:
            geometry.add_geo(sim)
        # add the index monitors

        self.fom.add_to_sim(sim)

        return sim

    def get_fom_geo(self, geometry=None):

        '''Will make and run the simulation, extract the figure of merit and return it

        :param geometry: If None, the current geometry will be used
        :returns: fom, The figure of merit calculated'''

        # create the simulation
        sim = self.make_sim(geometry=geometry)
        # run the simulation
        sim.run(self.iteration)
        # get the fom
        fom = self.fom.get_fom(sim)
        sim.close()
        return fom

    def run_forward_solves(self, plotfields=False):
        '''
        Generates the new forward simulations, runs them and computes the figure of merit and forward fields

        :param plotfields: Will plot the fields if True

        Since this assumes that this is used only in an optimization loop, the figure of merit is recorded and appended
        to the fomHist
        '''

        print('Running forward solves')

        # create the simulation
        forward_sim = self.make_sim()

        # run the simulation
        forward_sim.run(name = 'forward', iter = self.optimizer.iteration)

        # get the fields used for gradient calculation
        self.forward_fields = forward_sim.get_gradient_fields('opt_fields')

        # get the fom
        fom = self.fom.get_fom(forward_sim)
        self.fomHist.append(fom)

        forward_sim.close()

        print('FOM = {}'.format(fom))
        return fom

    def run_adjoint_solves(self, plotfields=False):
        '''
        Generates the adjoint simulations, runs them and extacts the adjoint fields
        '''

        print('Running adjoint solves')

        adjoint_sim = self.make_sim()

        # Remove the forward sources and add the adjoint sources
        adjoint_sim.remove_sources()
        self.fom.add_adjoint_sources(adjoint_sim)

        adjoint_sim.run(name = 'adjoint', iter = self.optimizer.iteration)

        self.adjoint_fields = adjoint_sim.get_gradient_fields('opt_fields')
        self.adjoint_fields.scale(3, self.fom.get_adjoint_field_scaling(adjoint_sim))
        adjoint_sim.close()


    def callable_fom(self,params):
        '''A callable function for the optimizers for the figure of merit
        :param params: The geometry parameters

        :returns: the fom
        '''
        self.geometry.update_geometry(params)
        fom = self.run_forward_solves()
        return fom

    def callable_jac(self,params):
        '''A callable function for the optimizer that returns derivatives with respect to the parameters

        :param params: The geometry paramaters, but actually these aren't used
        :returns: The gradients'''

        self.run_adjoint_solves()
        gradients = self.calculate_gradients()
        return np.array(gradients)

    def calculate_gradients(self,real=True):
        '''Uses the forward and adjoint fields to calculate the derivatives to the optimization parameters
            Assumes the forward and adjoint solves have been run'''

        print('Calculating gradients')

        # Create the gradient fields
        self.gradient_fields = Gradient_fields(forward_fields=self.forward_fields, adjoint_fields=self.adjoint_fields)

        'Let the geometry calculate the actual gradients'
        if not self.use_deps:
            self.gradients = self.geometry.calculate_gradients(self.gradient_fields, self.fom.wavelengths,real=real)
        else:
            self.gradients = self.geometry.calculate_gradients_from_sims_eps(self.gradient_fields,real=real)
        return self.gradients

    @staticmethod
    def goto_new_opts_folder(calling_file_name, base_script):
        ''' Creates a new folder in the current working directory named opt_xx to store the project files of the
            various simulations run during the optimization. Backup copiesof the calling and base scripts are 
            placed in the new folder.'''

        calling_file_path = os.path.dirname(calling_file_name) if os.path.isfile(calling_file_name) else os.path.dirname(os.getcwd())
        calling_file_path_entries = os.listdir(calling_file_path)
        opts_dir_numbers = [int(entry.split('_')[-1]) for entry in calling_file_path_entries if entry.startswith('opts_')]
        opts_dir_numbers.append(-1)
        new_opts_dir = os.path.join(calling_file_path, 'opts_{}'.format(max(opts_dir_numbers) + 1))
        os.mkdir(new_opts_dir)
        os.chdir(new_opts_dir)
        if os.path.isfile(calling_file_name):
            shutil.copy(calling_file_name, new_opts_dir)
        with open('script_file.lsf','a') as file:
            file.write(base_script.replace(';',';\n'))
