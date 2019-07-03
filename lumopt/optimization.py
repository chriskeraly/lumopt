""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import os
import shutil
import inspect
import copy
import numpy as np
import matplotlib.pyplot as plt

from lumopt.utilities.base_script import BaseScript
from lumopt.utilities.wavelengths import Wavelengths
from lumopt.utilities.simulation import Simulation
from lumopt.utilities.fields import FieldsNoInterp
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
            self.optimizer.initialize(start_params=start_params,
                                      callable_fom=callable_fom,
                                      callable_jac=callable_jac,
                                      bounds=bounds,
                                      plotting_function=plotting_function)

    def run(self):
        self.initialize()
        self.plotter=self.optimizations[0].init_plotter()

        if self.plotter.movie:
            with self.plotter.writer.saving(self.plotter.fig, "optimization.mp4", 100):
                self.optimizer.run()
        else:
            self.optimizer.run()

        final_fom = np.abs(self.optimizer.fom_hist[-1])
        print('FINAL FOM = {}'.format(final_fom))
        print('FINAL PARAMETERS = {}'.format(self.optimizer.params_hist[-1]))
        return final_fom,self.optimizer.params_hist[-1]


class Optimization(SuperOptimization):
    """ Acts as orchestrator for all the optimization pieces. Calling the member function run will perform the optimization,
        which requires four key pieces: 
            1) a script to generate the base simulation,
            2) an object that defines and collects the figure of merit,
            3) an object that generates the shape under optimization for a given set of optimization parameters and
            4) a gradient based optimizer.

        Parameters
        ----------
        :param base_script:    callable, file name or plain string with script to generate the base simulation.
        :param wavelengths:    wavelength value (float) or range (class Wavelengths) with the spectral range for all simulations.
        :param fom:            figure of merit (class ModeMatch).
        :param geometry:       optimizable geometry (class FunctionDefinedPolygon).
        :param optimizer:      SciyPy minimizer wrapper (class ScipyOptimizers).
        :param hide_fdtd_cad:  flag run FDTD CAD in the background.
        :param use_deps:       flag to use the numerical derivatives calculated directly from FDTD.
        :param plot_history:   plot the history of all parameters (and gradients)
        :param store_all_simulations: Indicates if the project file for each iteration should be stored or not  
    """

    def __init__(self, base_script, wavelengths, fom, geometry, optimizer, use_var_fdtd = False, hide_fdtd_cad = False, use_deps = True, plot_history = True, store_all_simulations = True):
        self.base_script = base_script if isinstance(base_script, BaseScript) else BaseScript(base_script)
        self.wavelengths = wavelengths if isinstance(wavelengths, Wavelengths) else Wavelengths(wavelengths)
        self.fom = fom
        self.geometry = geometry
        self.optimizer = optimizer
        self.use_var_fdtd = bool(use_var_fdtd)
        self.hide_fdtd_cad = bool(hide_fdtd_cad)
        self.use_deps = bool(use_deps)
        self.plot_history = bool(plot_history)
        self.store_all_simulations = store_all_simulations
        self.unfold_symmetry = geometry.unfold_symmetry

        if self.use_deps:
            print("Accurate interface detection enabled")

        self.plotter = None #< Initialize later, when we know how many parameters there are
        self.fomHist = []
        self.paramsHist = []

        frame = inspect.stack()[1]
        calling_file_name = os.path.abspath(frame[0].f_code.co_filename)
        Optimization.goto_new_opts_folder(calling_file_name, base_script)
        self.workingDir = os.getcwd()

    def __del__(self):
        Optimization.go_out_of_opts_folder()

    def init_plotter(self):
        if self.plotter is None:
            self.plotter = Plotter(movie = True, plot_history = self.plot_history)
        return self.plotter

    def run(self):
        self.initialize()

        self.init_plotter()
        
        if self.plotter.movie:
            with self.plotter.writer.saving(self.plotter.fig, "optimization.mp4", 100):
                self.optimizer.run()
        else:
            self.optimizer.run()

        ## For topology optimization we are not done yet ... 
        if hasattr(self.geometry,'progress_continuation'):
            print(' === Starting Binarization Phase === ')
            self.optimizer.max_iter=20
            while self.geometry.progress_continuation():
                self.optimizer.reset_start_params(self.optimizer.params_hist[-1], 0.05) #< Run the scaling analysis again
                self.optimizer.run()

        final_fom = np.abs(self.optimizer.fom_hist[-1])
        print('FINAL FOM = {}'.format(final_fom))
        print('FINAL PARAMETERS = {}'.format(self.optimizer.params_hist[-1]))
        return final_fom,self.optimizer.params_hist[-1]

    def initialize(self):
        """ 
            Performs all steps that need to be carried only once at the beginning of the optimization. 
        """

        # FDTD CAD
        self.sim = Simulation(self.workingDir, self.use_var_fdtd, self.hide_fdtd_cad)
        # FDTD model
        self.base_script(self.sim.fdtd)
        Optimization.set_global_wavelength(self.sim, self.wavelengths)
        Optimization.set_source_wavelength(self.sim, 'source', self.fom.multi_freq_src, len(self.wavelengths))
        self.sim.fdtd.setnamed('opt_fields', 'override global monitor settings', False)
        self.sim.fdtd.setnamed('opt_fields', 'spatial interpolation', 'none')
        Optimization.add_index_monitor(self.sim, 'opt_fields')
        
        if self.use_deps:
            Optimization.set_use_legacy_conformal_interface_detection(self.sim, False)

        # Optimizer
        start_params = self.geometry.get_current_params()

        # We need to add the geometry first because it adds the mesh override region
        self.geometry.add_geo(self.sim, start_params, only_update = False)

        # If we don't have initial parameters yet, try to extract them from the simulation (this is mostly for topology optimization)
        if start_params is None:
            self.geometry.extract_parameters_from_simulation(self.sim)
            start_params = self.geometry.get_current_params()

        callable_fom = self.callable_fom
        callable_jac = self.callable_jac
        bounds = np.array(self.geometry.bounds)

        def plotting_function():
            self.plotter.update(self)
            
            if hasattr(self.geometry,'to_file'):
                self.geometry.to_file('parameters_{}.npz'.format(self.optimizer.iteration))

            with open('convergence_report.txt','a') as f:
                f.write('{}, {}'.format(self.optimizer.iteration,self.optimizer.fom_hist[-1]))
                if hasattr(self.geometry,'write_status'):
                    self.geometry.write_status(f) 
                f.write('\n')


        self.fom.initialize(self.sim)

        self.optimizer.initialize(start_params = start_params, callable_fom = callable_fom, callable_jac = callable_jac, bounds = bounds, plotting_function = plotting_function)
      

    def make_forward_sim(self, params):
        self.sim.fdtd.switchtolayout()
        self.geometry.update_geometry(params, self.sim)
        self.geometry.add_geo(self.sim, params = None, only_update = True)
        self.sim.fdtd.setnamed('source', 'enabled', True)
        self.fom.make_forward_sim(self.sim)

    def run_forward_solves(self, params):
        """ Generates the new forward simulations, runs them and computes the figure of merit and forward fields. """

        print('Running forward solves')
        self.make_forward_sim(params)
        iter = self.optimizer.iteration if self.store_all_simulations else 0
        self.sim.run(name = 'forward', iter = iter)
       
        get_eps = True
        get_D = not self.use_deps
        nointerpolation = not self.geometry.use_interpolation()
        
        self.forward_fields = get_fields(self.sim.fdtd,
                                         monitor_name = 'opt_fields',
                                         field_result_name = 'forward_fields',
                                         get_eps = get_eps,
                                         get_D = get_D,
                                         get_H = False,
                                         nointerpolation = nointerpolation,
                                         unfold_symmetry = self.unfold_symmetry)
        fom = self.fom.get_fom(self.sim)

        if self.store_all_simulations:
            self.sim.remove_data_and_save() #< Remove the data from the file to save disk space. TODO: Make optional?

        self.fomHist.append(fom)
        print('FOM = {}'.format(fom))
        return fom

    def make_adjoint_sim(self, params):
        self.sim.fdtd.switchtolayout()
        assert np.allclose(params, self.geometry.get_current_params())
        self.geometry.add_geo(self.sim, params = None, only_update = True)
        self.sim.fdtd.setnamed('source', 'enabled', False)
        self.fom.make_adjoint_sim(self.sim)

    def run_adjoint_solves(self, params):
        """ Generates the adjoint simulations, runs them and extacts the adjoint fields. """

        has_forward_fields = hasattr(self,'forward_fields') and hasattr(self.forward_fields, 'E')
        params_changed = not np.allclose(params, self.geometry.get_current_params())
        if not has_forward_fields or params_changed:
            fom = self.run_forward_solves(params)

        print('Running adjoint solves')
        self.make_adjoint_sim(params)

        iter = self.optimizer.iteration if self.store_all_simulations else 0
        self.sim.run(name = 'adjoint', iter = iter)
        
        get_eps = not self.use_deps
        get_D = not self.use_deps
        nointerpolation = not self.geometry.use_interpolation()

        #< JN: Try on CAD
        self.adjoint_fields = get_fields(self.sim.fdtd,
                                         monitor_name = 'opt_fields',
                                         field_result_name = 'adjoint_fields',
                                         get_eps = get_eps,
                                         get_D = get_D,
                                         get_H = False,
                                         nointerpolation = nointerpolation,
                                         unfold_symmetry = self.unfold_symmetry)
        self.adjoint_fields.scaling_factor = self.fom.get_adjoint_field_scaling(self.sim)

        self.adjoint_fields.scale(3, self.adjoint_fields.scaling_factor)

        if self.store_all_simulations:
            self.sim.remove_data_and_save() #< Remove the data from the file to save disk space. TODO: Make optional?

    def callable_fom(self, params):
        """ Function for the optimizers to retrieve the figure of merit.
            :param params:  optimization parameters.
            :param returns: figure of merit.
        """
        return self.run_forward_solves(params)

    def callable_jac(self, params):
        """ Function for the optimizer to extract the figure of merit gradient.
            :param params:  optimization paramaters.
            :param returns: partial derivative of the figure of merit with respect to each optimization parameter.
        """
        self.run_adjoint_solves(params)
        return self.calculate_gradients()

    def calculate_gradients(self):
        """ Calculates the gradient of the figure of merit (FOM) with respect to each of the optimization parameters.
            It assumes that both the forward and adjoint solves have been run so that all the necessary field results
            have been collected. There are currently two methods to compute the gradient:
                1) using the permittivity derivatives calculated directly from meshing (use_deps == True) and
                2) using the shape derivative approximation described in Owen Miller's thesis (use_deps == False).
        """

        print('Calculating gradients')
        fdtd = self.sim.fdtd
        self.gradient_fields = GradientFields(forward_fields = self.forward_fields, adjoint_fields = self.adjoint_fields)
        self.sim.fdtd.switchtolayout()
        if self.use_deps:
            self.geometry.d_eps_on_cad(self.sim)
            fom_partial_derivs_vs_wl = GradientFields.spatial_gradient_integral_on_cad(self.sim, 'forward_fields', 'adjoint_fields', self.adjoint_fields.scaling_factor)
            self.gradients = self.fom.fom_gradient_wavelength_integral(fom_partial_derivs_vs_wl.transpose(), self.forward_fields.wl)
        else:
            if hasattr(self.geometry,'calculate_gradients_on_cad'):
                fom_partial_derivs_vs_wl = self.geometry.calculate_gradients_on_cad(self.sim, 'forward_fields', 'adjoint_fields', self.adjoint_fields.scaling_factor)
                self.gradients = self.fom.fom_gradient_wavelength_integral(fom_partial_derivs_vs_wl, self.forward_fields.wl)
            else:
                fom_partial_derivs_vs_wl = self.geometry.calculate_gradients(self.gradient_fields)
                self.gradients = self.fom.fom_gradient_wavelength_integral(fom_partial_derivs_vs_wl, self.forward_fields.wl)
        return self.gradients

    @staticmethod
    def goto_new_opts_folder(calling_file_name, base_script):
        ''' Creates a new folder in the current working directory named opt_xx to store the project files of the
            various simulations run during the optimization. Backup copiesof the calling and base scripts are 
            placed in the new folder.'''

        calling_file_path = os.path.dirname(calling_file_name) if os.path.isfile(calling_file_name) else os.path.dirname(os.getcwd())
        calling_file_path_split = os.path.split(calling_file_path)
        if calling_file_path_split[1].startswith('opts_'):
            calling_file_path = calling_file_path_split[0]
        calling_file_path_entries = os.listdir(calling_file_path)
        opts_dir_numbers = [int(entry.split('_')[-1]) for entry in calling_file_path_entries if entry.startswith('opts_')]
        opts_dir_numbers.append(-1)
        new_opts_dir = os.path.join(calling_file_path, 'opts_{}'.format(max(opts_dir_numbers) + 1))
        os.mkdir(new_opts_dir)
        os.chdir(new_opts_dir)
        if os.path.isfile(calling_file_name):
            shutil.copy(calling_file_name, new_opts_dir)
        if hasattr(base_script, 'script_str'):
            with open('script_file.lsf','a') as file:
                file.write(base_script.script_str.replace(';',';\n'))

    @staticmethod
    def go_out_of_opts_folder():
        cwd_split = os.path.split(os.path.abspath(os.getcwd()))
        if cwd_split[1].startswith('opts_'):
            os.chdir(cwd_split[0])

    @staticmethod
    def add_index_monitor(sim, monitor_name):
        sim.fdtd.select(monitor_name)
        if sim.fdtd.getnamednumber(monitor_name) != 1:
            raise UserWarning("a single object named '{}' must be defined in the base simulation.".format(monitor_name))
        index_monitor_name = monitor_name + '_index'
        if sim.fdtd.getnamednumber('FDTD') == 1:
            sim.fdtd.addindex()
        elif sim.fdtd.getnamednumber('varFDTD') == 1:
            sim.fdtd.addeffectiveindex()
        else:
            raise UserWarning('no FDTD or varFDTD solver object could be found.')
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
    def set_source_wavelength(sim, source_name, multi_freq_src, freq_pts):
        if sim.fdtd.getnamednumber(source_name) != 1:
            raise UserWarning("a single object named '{}' must be defined in the base simulation.".format(source_name))
        if sim.fdtd.getnamed(source_name, 'override global source settings'):
            print('Wavelength range of source object will be superseded by the global settings.')
        sim.fdtd.setnamed(source_name, 'override global source settings', False)
        sim.fdtd.select(source_name)
        if sim.fdtd.haveproperty('multifrequency mode calculation'):
            sim.fdtd.setnamed(source_name, 'multifrequency mode calculation', multi_freq_src)
            if multi_freq_src:
                sim.fdtd.setnamed(source_name, 'frequency points', freq_pts)
        elif sim.fdtd.haveproperty('multifrequency beam calculation'):
            sim.fdtd.setnamed(source_name, 'multifrequency beam calculation', multi_freq_src)
            if multi_freq_src:
                sim.fdtd.setnamed(source_name, 'number of frequency points', freq_pts)

    @staticmethod
    def set_use_legacy_conformal_interface_detection(sim, flagVal):
        if sim.fdtd.getnamednumber('FDTD') == 1:
            sim.fdtd.select('FDTD')
        elif sim.fdtd.getnamednumber('varFDTD') == 1:
            sim.fdtd.select('varFDTD')
        else:
            raise UserWarning('no FDTD or varFDTD solver object could be found.')
        if bool(sim.fdtd.haveproperty('use legacy conformal interface detection')):
            sim.fdtd.set('use legacy conformal interface detection', flagVal)
            sim.fdtd.set('conformal meshing refinement', 51)
            sim.fdtd.set('meshing tolerance', 1.0/1.134e14)
        else:
            raise UserWarning('install a more recent version of FDTD or the permittivity derivatives will not be accurate.')
            
