""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import os
import time
import shutil
import inspect
import copy
import numpy as np
import matplotlib.pyplot as plt

from lumopt.utilities.wavelengths import Wavelengths
from lumopt.utilities.simulation import Simulation
from lumopt.utilities.gradients import GradientFields
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.utilities.plotter import Plotter
from lumopt.lumerical_methods.lumerical_scripts import get_fields

class SuperOptimization(object):
    """
        Optimization super class to run two or more co-optimizations targeting different figures of merit that take the same parameters.
        The addition operator can be used to aggregate multiple optimizations. All the figures of merit are simply added to generate 
        an overall figure of merit that is passed to the chosen optimizer.

        Parameters
        ----------
        :param optimizations: list of co-optimizations (each of class Optimization). 
    """

    def __init__(self,optimizations):
        self.optimizations=optimizations

    def __add__(self,other):
        optimizations=[self,other]
        return SuperOptimization(optimizations)

    def initialize(self,start_params=None,bounds=None):

        print('Initializing super optimization')

        self.optimizer = copy.deepcopy(self.optimizations[0].optimizer)
        self.plotter=self.optimizations[0].plotter

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
        self.initialize()
        if self.plotter.movie:
            with self.plotter.writer.saving(self.plotter.fig, "optimization.mp4", 100):
                self.optimizer.run()
        else:
            self.optimizer.run()

        print('FINAL FOM = {}'.format(self.optimizer.fom_hist[-1]))
        print('FINAL PARAMETERS = {}'.format(self.optimizer.params_hist[-1]))
        return self.optimizer.fom_hist[-1],self.optimizer.params_hist[-1]


class Optimization(SuperOptimization):
    """ Acts as orchestrator for all the optimization pieces. Calling the member function run will perform the optimization,
        which requires four key pieces: 
            1) a script to generate the base simulation,
            2) an object that defines and collects the figure of merit,
            3) an object that generates the shape under optimization for a given set of optimization parameters and
            4) a gradient based optimizer.

        Parameters
        ----------
        :param base_script: string with script to generate the base simulation (helper function load_from_lsf).
        :param wavelengths: wavelength value (float) or range (class Wavelengths) with the spectral range for all simulations.
        :param fom:         figure of merit (class ModeMatch).
        :param geometry:    optimizable geometry (class FunctionDefinedPolygon).
        :param optimizer:   SciyPy minimizer wrapper (class ScipyOptimizers).
        :param hide_fdtd:   flag run FDTD CAD in the background.
        :param use_deps:    flag to use the numerical derivatives calculated directly from FDTD.
    """

    def __init__(self, base_script, wavelengths, fom, geometry, optimizer, hide_fdtd_cad = False, use_deps = True):
        self.base_script = base_script
        self.wavelengths = wavelengths if isinstance(wavelengths, Wavelengths) else Wavelengths(wavelengths)
        self.fom = fom
        self.geometry = geometry
        self.optimizer = optimizer
        self.hide_fdtd_cad = bool(hide_fdtd_cad)
        self.use_deps = bool(use_deps)
        if self.use_deps:
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
        """ Performs all steps that need to be carried only once at the beginning of the optimization. """

        start_params = self.geometry.get_current_params()
        callable_fom = self.callable_fom
        callable_jac = self.callable_jac
        bounds = np.array(self.geometry.bounds)
        def plotting_function():
            self.plotter.update(self)
        self.optimizer.initialize(start_params = start_params, callable_fom = callable_fom, callable_jac = callable_jac, bounds = bounds, plotting_function = plotting_function)
        self.sim = Simulation(self.workingDir, self.hide_fdtd_cad)

    def make_forward_sim(self, geometry = None):
        """ Creates the forward simulation by adding the geometry to the base simulation and adding a refractive index monitor overlaping
            with the 'opt_fields' monitor. The 'source' object is modified to follow the global frequency settings.

            :geometry: the current gometry under optimization.
        """

        self.sim.fdtd.switchtolayout()
        self.sim.fdtd.deleteall()
        self.sim.fdtd.eval(self.base_script)
        Optimization.set_global_wavelength(self.sim, self.wavelengths)
        Optimization.set_source_wavelength(self.sim, 'source', len(self.wavelengths))
        self.sim.fdtd.setnamed('opt_fields', 'override global monitor settings', False)
        self.sim.fdtd.setnamed('opt_fields', 'spatial interpolation', 'none')
        Optimization.add_index_monitor(self.sim, 'opt_fields')
        if self.use_deps:
            Optimization.set_use_legacy_conformal_interface_detection(self.sim, False)
        time.sleep(0.1)
        if geometry is None:
            self.geometry.add_geo(self.sim, params = None, only_update = False)
        else:
            geometry.add_geo(self.sim)
        self.fom.add_to_sim(self.sim)

    def run_forward_solves(self):
        """ Generates the new forward simulations, runs them and computes the figure of merit and forward fields. """

        print('Running forward solves')
        self.make_forward_sim()
        self.sim.run(name = 'forward', iter = self.optimizer.iteration)
        self.forward_fields = get_fields(self.sim.fdtd, monitor_name = 'opt_fields', get_eps = True, get_D = True, get_H = True, nointerpolation = True)
        fom = self.fom.get_fom(self.sim)
        self.fomHist.append(fom)
        print('FOM = {}'.format(fom))
        return fom

    def run_adjoint_solves(self):
        """ Generates the adjoint simulations, runs them and extacts the adjoint fields. """

        print('Running adjoint solves')
        self.make_forward_sim()
        self.sim.fdtd.selectpartial('source')
        self.sim.fdtd.delete()
        self.fom.add_adjoint_sources(self.sim)
        self.sim.run(name = 'adjoint', iter = self.optimizer.iteration)
        self.adjoint_fields = get_fields(self.sim.fdtd, monitor_name = 'opt_fields', get_eps = True, get_D = True, get_H = True, nointerpolation = True)
        self.adjoint_fields.scale(3, self.fom.get_adjoint_field_scaling(self.sim))

    def callable_fom(self, params):
        """ Function for the optimizers to retrieve the figure of merit.
            :param params:  geometry parameters.
            :returns: figure of merit.
        """

        self.geometry.update_geometry(params)
        fom = self.run_forward_solves()
        return fom

    def callable_jac(self, params):
        """ Function for the optimizer to extract the figure of merit gradient.
            :params:  geometry paramaters, currently not used.
            :returns: partial derivative of the figure of merit with respect to each optimization parameter.
        """

        self.run_adjoint_solves()
        gradients = self.calculate_gradients()
        return np.array(gradients)

    def calculate_gradients(self):
        """ Calculates the gradient of the figure of merit (FOM) with respect to each of the optimization parameters.
            It assumes that both the forward and adjoint solves have been run so that all the necessary field results
            have been collected. There are currently two methods to compute the gradient:
                1) using the permittivity derivatives calculated directly from meshing (use_deps == True) and
                2) using the shape derivative approximation described in Owen Miller's thesis (use_deps == False).
        """

        print('Calculating gradients')
        self.gradient_fields = GradientFields(forward_fields = self.forward_fields, adjoint_fields = self.adjoint_fields)
        if self.use_deps:
            self.make_forward_sim()
            d_eps = self.geometry.get_d_eps(self.sim)
            fom_partial_derivs_vs_wl, wl = self.gradient_fields.spatial_gradient_integral(d_eps)
            self.gradients = self.fom.fom_gradient_wavelength_integral(fom_partial_derivs_vs_wl, wl)
        else:
            fom_partial_derivs_vs_wl = self.geometry.calculate_gradients(self.gradient_fields)
            wl = self.gradient_fields.forward_fields.wl
            self.gradients = self.fom.fom_gradient_wavelength_integral(fom_partial_derivs_vs_wl.transpose(), wl)
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

    @staticmethod
    def add_index_monitor(sim, monitor_name):
        sim.fdtd.select(monitor_name)
        if sim.fdtd.getnamednumber(monitor_name) != 1:
            raise UserWarning("a single object named '{}' must be defined in the base simulation.".format(monitor_name))
        index_monitor_name = monitor_name + '_index'
        sim.fdtd.addindex()
        sim.fdtd.set('name', index_monitor_name)
        sim.fdtd.setnamed(index_monitor_name, 'override global monitor settings', True)
        sim.fdtd.setnamed(index_monitor_name, 'frequency points', 1)
        sim.fdtd.setnamed(index_monitor_name, 'record conformal mesh when possible', True)
        monitor_type = sim.fdtd.getnamed(monitor_name, 'monitor type')
        geometric_props = ['monitor type']
        geometric_props.extend(Optimization.cross_section_monitor_props(monitor_type))
        for prop_name in geometric_props:
            prop_val = sim.fdtd.getnamed(monitor_name, prop_name)
            sim.fdtd.setnamed(index_monitor_name, prop_name, prop_val)
        sim.fdtd.setnamed(index_monitor_name, 'spatial interpolation', 'none')

    @staticmethod
    def cross_section_monitor_props(monitor_type):
        geometric_props = ['x', 'y', 'z']
        if monitor_type == '3D':
            geometric_props.extend(['x span','y span','z span'])
        elif monitor_type == '2D X-normal':
            geometric_props.extend(['y span','z span'])
        elif monitor_type == '2D Y-normal':
            geometric_props.extend(['x span','z span'])
        elif monitor_type == '2D Z-normal':
            geometric_props.extend(['x span','y span'])
        elif monitor_type == 'Linear X':
            geometric_props.append('x span')
        elif monitor_type == 'Linear Y':
            geometric_props.append('y span')
        elif monitor_type == 'Linear Z':
            geometric_props.append('z span')
        else:
            raise UserWarning('monitor should be 2D or linear for a mode expansion to be meaningful.')
        return geometric_props

    @staticmethod
    def set_global_wavelength(sim, wavelengths):
        sim.fdtd.setglobalmonitor('use source limits', True)
        sim.fdtd.setglobalmonitor('use linear wavelength spacing', True)
        sim.fdtd.setglobalmonitor('frequency points', len(wavelengths))
        sim.fdtd.setglobalsource('set wavelength', True)
        sim.fdtd.setglobalsource('wavelength start', wavelengths.min())
        sim.fdtd.setglobalsource('wavelength stop', wavelengths.max())

    @staticmethod
    def set_source_wavelength(sim, source_name, freq_pts):
        if sim.fdtd.getnamednumber(source_name) != 1:
            raise UserWarning("a single object named '{}' must be defined in the base simulation.".format(source_name))
        if sim.fdtd.getnamed(source_name, 'override global source settings'):
            print('Wavelength range of source object will be superseded by the global settings.')
        sim.fdtd.setnamed(source_name, 'override global source settings', False)
        sim.fdtd.select(source_name)
        if sim.fdtd.haveproperty('multifrequency mode calculation'):
            sim.fdtd.setnamed(source_name, 'multifrequency mode calculation', True)
            sim.fdtd.setnamed(source_name, 'frequency points', freq_pts)
        elif sim.fdtd.haveproperty('multifrequency beam calculation'):
            sim.fdtd.setnamed(source_name, 'multifrequency beam calculation', True)
            sim.fdtd.setnamed(source_name, 'number of frequency points', freq_pts)
        else:
            raise UserWarning('unable to determine source type.')

    @staticmethod
    def set_use_legacy_conformal_interface_detection(sim, flagVal):
        sim.fdtd.select('FDTD')
        has_legacy_prop = bool(sim.fdtd.haveproperty('use legacy conformal interface detection'))
        if has_legacy_prop:
            sim.fdtd.setnamed('FDTD', 'use legacy conformal interface detection', flagVal)
            sim.fdtd.setnamed('FDTD', 'conformal meshing refinement', 51)
        else:
            raise UserWarning('install a more recent version of FDTD or the permittivity derivatives will not be accurate.')
