from lumopt.lumerical_methods.lumerical_scripts import add_D_monitors_to_fields_monitors, remove_interpolation_on_monitor, add_index_to_fields_monitors, set_spatial_interp
from lumopt.utilities.simulation import Simulation
from lumopt.utilities.gradients import Gradient_fields
from time import sleep
import copy
from lumopt.utilities.plotter import Plotter
from lumopt.utilities.scipy_wrappers import trapz3D
import numpy as np
import os
from copy import deepcopy
import matplotlib.pyplot as plt

class Super_Optimization(object):
    ''' Optimization parent class which allows the user to use the + operator to co-optimize two optimizations which use
    the same parameters. The figures of merit are simply added. Plotting functions for fields and geometries uses the first
    optimization'''

    def __init__(self,optimizations):
        self.optimizations=optimizations

    def __add__(self,other):
        optimizations=[self,other]
        return Super_Optimization(optimizations)

    def initialize(self,start_params=None,bounds=None):

        print('Initializing Super Optimization')

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


    def run(self, verbose=True):
        '''Inititalizes and then runs the optimization

        :param verbose: Defines how verbose the optimization should be'''

        self.verbose = verbose
        self.initialize()
        if self.plotter.movie:
            with self.plotter.writer.saving(self.plotter.fig, "optimization.mp4", 100):
                self.optimizer.run(verbose)
        else:
            self.optimizer.run(verbose)

        print('FINAL FOM = {}'.format(self.optimizer.fom_hist[-1]))
        print('FINAL PARAMETERS = {}'.format(self.optimizer.params_hist[-1]))
        return self.optimizer.fom_hist[-1],self.optimizer.params_hist[-1]


class Optimization(Super_Optimization):
    '''The Optimization class contains all the aspects of the optimization problem, and can ultimitely initialize and run
    the optimization once it has been set up.\n\n
    In order for the optimization to run, it needs to know:
        - How to build the forward problem
        - What the figure of merit is (FOM)
        - What the geometry to optimize is, and what are it's shape parameters
        - What optimization subroutine to use (Gradient descent, l-BFGS-G...)

    '''


    def __init__(self, base_script, fom, geometry, optimizer, plotter=Plotter(),use_deps=False):
        '''
        :param script:
            A lumerical script that can build the underalying base for the forward problem. There is helper function to load it from an lsf file (:func:`lumopt.utilities.load_lumerical_scripts.load_from_lsf`)
        :param fom:
            an instance of a FOM class. This class knows how to calculate the figure of merit from fields in the simulation
            as well as what the adjoint sources should be
        :param geometry:
            an instance of a geometry class. The geometry is usually parametrized by a set of parameters which
            are the parameters to optimize. The geometry class knows how to add itself to a simulation, and how to calculate the
            derivatives of the figure of merit wrt to it's parameters, given the forward and adjoint fields.
        :param optimizer:
            An instance of an optimizer class, it will decide from the figures of merits and the Jacobians calculated how to
            update the shape parameters
        :param Plotter:
            Plotter object. Is set by default and shouldn't need tweaking.
        :param use_deps:
            In development. This will implement a discrete adjoint in space, by directly extracting the permittivity derivatives
            from Lumerical. There are changes that need to be implemented within Lumerical for this to work.

        '''

        self.base_script = base_script
        self.fom = fom
        self.geometry = geometry
        self.optimizer = optimizer
        self.plotter=plotter
        self.forward_fields = None
        self.adjoint_fields = None
        self.gradients = None
        self.fomHist = []
        self.paramsHist = []
        self.iteration = 0
        self.gradient_fields = None
        self.verbose = True
        self.use_deps = use_deps

        self.goto_new_sims_folder()
        self.workingDir = os.getcwd()

        '''TODO: this class should know something about the variables it is optimizing, I don't think it should
        only be the geometry'''

    def run(self, verbose=True):
        '''Inititalizes and then runs the optimization

        :param verbose: Defines how verbose the optimization should be'''

        self.verbose = verbose
        self.initialize()
        if self.plotter.movie:
            with self.plotter.writer.saving(self.plotter.fig, "optimization.mp4", 100):
                self.optimizer.run(verbose)

        else:
            self.optimizer.run(verbose)

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
        if hasattr(self.fom,'initialize'):
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
        '''Creates the substrate simulation without the optimizable geometry'''
        base_sim = Simulation(workingDir=self.workingDir,script=self.base_script)
        return base_sim

    def make_sim(self, geometry=None,put_monitors=True):
        '''Creates the forward simulation by adding the geometry to the base simulation and adding D monitors to any field monitor in the simulation.
        If the FOM object needs it's own monitors (or needs to manipulate those already present) they will also be added/manipulated.

        :param geometry: By default the current gometry of the Optimization #will be put in, but this can be overriden by inputing another geometry here
        :param put_monitors: Whether or not the fom monitors should try to be added

        :returns: sim, Handle to the simulation '''

        # create the simulation object
        sim = self.make_base_sim()
        #add_D_monitors_to_fields_monitors(sim.solver_handle, 'opt_fields')
        add_index_to_fields_monitors(sim.fdtd, 'opt_fields')
        set_spatial_interp(sim.fdtd,'opt_fields','None')

        # add the optimizable geometry
        if geometry is None:
            self.geometry.add_geo(sim)
        else:
            geometry.add_geo(sim)
        # add the index monitors

        sleep(0.1)

        #remove_interpolation_on_monitor(sim.solver_handle,'opt_fields')
        if put_monitors:
            try:
                self.fom.put_monitors(sim)
            except:
                pass #print('Could not add fom monitors')
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

        if self.verbose:
            print('Running Forward Solves')

        # create the simulation
        forward_sim = self.make_sim()

        # run the simulation
        forward_sim.run(name='forward',iter=self.iteration)

        # get the fields used for gradient calculation
        self.forward_fields = forward_sim.get_gradient_fields('opt_fields')

        # get the fom
        fom = self.fom.get_fom(forward_sim)
        self.fomHist.append(fom)

        forward_sim.close()

        if self.verbose:
            print('FOM={}'.format(fom))
        return fom

    def run_adjoint_solves(self, plotfields=False):
        '''
        Generates the adjoint simulations, runs them and extacts the adjoint fields
        '''
        if self.verbose:
            print('Running adjoint Solves')

        adjoint_sim = self.make_sim()

        # Remove the forward sources and add the adjoint sources
        adjoint_sim.remove_sources()
        self.fom.add_adjoint_sources(adjoint_sim)

        adjoint_sim.run(name='adjoint',iter=self.iteration)

        self.adjoint_fields = adjoint_sim.get_gradient_fields('opt_fields')
        adjoint_sim.close()


    def make_adjoint_solves(self, sleep_time=1000):
        '''
        Generates the adjoint simulations, moslty for testing
        '''
        if self.verbose:
            print('Running adjoint Solves')

        adjoint_sim = self.make_sim()

        # Remove the forward sources and add the adjoint sources
        adjoint_sim.remove_sources()
        self.fom.add_adjoint_sources(adjoint_sim)

        sleep(sleep_time)
        return

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

    from copy import deepcopy

    def calculate_finite_differences_gradients_2(self, n_derivatives=range(4, 6), dx=0.01e-9, central=False, print_res=True,
                                                 superverbose=False):

        '''Calculates the finite difference gradients, and also compares the derivative to the gradients calculated using the adjoint
        derivatives, as well as recalculated derivatives using the actual permittivity change seen in the simulation and extracted from the
        index monitors'''

        finite_differences_gradients = []
        recalculated_adjoint_derivs = []
        eps0 = 8.854e-12
        params = np.array(self.geometry.get_current_params())
        # if superverbose:print('Current parameters={}'.format(params))
        self.run_forward_solves()
        self.run_adjoint_solves()
        adjoint_gradients=self.calculate_gradients(real=False)
        print("Adjoint gradients= {}".format(adjoint_gradients))
        current_fom = self.fomHist[-1]
        current_eps = deepcopy(self.forward_fields.eps.copy())
        current_E = deepcopy(self.forward_fields.E)
        current_E_adj = deepcopy(self.adjoint_fields.E)
        sparse_pert_E = 2*eps0*current_E*current_E_adj
        if superverbose:
            print('Nominal FOM = {}'.format(current_fom))
        for i, param in zip(n_derivatives, params[n_derivatives]):
            if not central:
                # d_geo = copy.deepcopy(self.geometry)
                d_params = params.copy()
                d_params[i] = param + dx
                # d_geo.update_geometry(d_params)
                d_fom = self.callable_fom(d_params)
                if superverbose: print('dfom={}'.format(d_fom))
                deriv = (d_fom - current_fom)/dx
                finite_differences_gradients.append(deriv)
                d_eps = (self.forward_fields.eps - current_eps)/dx
                plt.pcolormesh(np.real(d_eps[:, :, 0, 0, 2]).transpose())
                plt.show()
                recalculated_adjoint_deriv = trapz3D(np.sum(sparse_pert_E*d_eps,axis=-1)[:,:,:,0],self.forward_fields.x,self.forward_fields.y,self.forward_fields.z)
                print('recalculated adjoint gradients={}'.format(recalculated_adjoint_deriv))
                recalculated_adjoint_derivs.append(recalculated_adjoint_deriv)
            else:
                print('central not supported on this on yet')

            if print_res: print('Derivative n {}={}'.format(i, deriv))
        self.geometry.update_geometry(params)

        if print_res:
            print(finite_differences_gradients)
            print(recalculated_adjoint_derivs)

        return finite_differences_gradients, recalculated_adjoint_derivs,adjoint_gradients

    def calculate_finite_differences_gradients(self, n_derivatives=range(4,6), dx=3e-9,central=True,print_res=True,superverbose=False):
        '''Calculates the derivatives using finite differences. This should be only used to verify the gradients

        :param n_derivatives: The number of derivatives that should be calculated (on to two simulations per derivative)
        :param dx: The finite step for derivative calculation
        :param central: Should a central finite difference scheme be used?
        :param print_res: guess :)

        :returns: finite_difference_gradients'''

        if self.geometry.self_update:
            finite_differences_gradients = self.geometry.calculate_finite_differences_gradients()
        else:
            finite_differences_gradients = []
            try: #If the geometry knows how to get it's own finite differences
                finite_differences_geometries = self.geometry.get_geometries_for_finite_differences_gradients(dx=dx)
                current_fom = self.get_fom_geo(self.geometry)
                for i, geometry in enumerate(finite_differences_geometries[:n_derivatives]):
                    fomdx = self.get_fom_geo(geometry)
                    deriv=(fomdx - current_fom)/dx
                    finite_differences_gradients.append(deriv)
                    if print_res: print('derivative n {}={}'.format(i, deriv))
            except:
                params=np.array(self.geometry.get_current_params())
                # if superverbose:print('Current parameters={}'.format(params))
                if not central or superverbose:
                    current_fom = self.get_fom_geo(self.geometry)
                    if superverbose:
                        print('Nominal FOM = {}'.format(current_fom))
                for i,param in zip(n_derivatives,params[n_derivatives]):
                    if not central:
                        # d_geo = copy.deepcopy(self.geometry)
                        d_params = params.copy()
                        d_params[i] = param + dx
                        # d_geo.update_geometry(d_params)
                        d_fom = self.callable_fom(d_params)
                        if superverbose: print('dfom={}'.format(d_fom))
                        deriv = (d_fom - current_fom)/dx
                        finite_differences_gradients.append(deriv)
                    else:
                        # d_geo_pos=copy.deepcopy(self.geometry)
                        # d_geo_neg=copy.deepcopy(self.geometry)
                        d_params_pos=params.copy()
                        d_params_neg=params.copy()
                        d_params_pos[i]=param+dx
                        d_params_neg[i]=param-dx
                        # d_geo_pos.update_geometry(d_params_pos)
                        # d_geo_neg.update_geometry(d_params_neg)
                        print('getting + '),
                        d_fom_pos=self.callable_fom(d_params_pos)
                        if superverbose: print('dfom + ={}'.format(d_fom_pos))
                        print('getting - '),
                        d_fom_neg=self.callable_fom(d_params_neg)
                        if superverbose: print('dfom + ={}'.format(d_fom_neg))
                        deriv=(d_fom_pos-d_fom_neg)/dx/2.
                        finite_differences_gradients.append(deriv)
                        # if superverbose: print('Current parameters={}'.format(self.geometry.get_current_params()))
                    if print_res: print('Derivative n {}={}'.format(i,deriv))
                self.geometry.update_geometry(params)

        if print_res:
            print(finite_differences_gradients)

        return finite_differences_gradients


    def calculate_gradients(self,real=True):
        '''Uses the forward and adjoint fields to calculate the derivatives to the optimization parameters
            Assumes the forward and adjoint solves have been run'''
        if self.verbose:
            print('Calculating Gradients')

        # Create the gradient fields
        self.gradient_fields = Gradient_fields(forward_fields=self.forward_fields, adjoint_fields=self.adjoint_fields)

        'Let the geometry calculate the actual gradients'
        if not self.use_deps:
            self.gradients = self.geometry.calculate_gradients(self.gradient_fields, self.fom.wavelengths,real=real)
        else:
            self.gradients = self.geometry.calculate_gradients_from_sims_eps(self.gradient_fields,real=real)


        return self.gradients

    def goto_new_sims_folder(self):
        '''Creates a new folder in the current working directory name opt_xx where xx can go up to 25 (an error is given if it goes above)
         This folder will store all the simulations of the optimization.'''

        ## THIS IS A TERRIBLE THING BUT FOR THE MOMENT I'LL USE IT
        # It's to copy the file that created the optimizer, which is probably a good chunk of the info to recreate it
        import inspect
        try:
            def get_caller(num):
                return inspect.stack()[num]  # 1 is get_caller's caller
            calling_file= os.path.abspath(inspect.getfile(get_caller(3)[0]))
        except:
            print('Couldnt copy python setupfile')


        directories = os.listdir(os.getcwd())
        try:
            old = max([int(x.split('_')[-1]) for x in directories if x.startswith('opts_')])
        except ValueError:  # in case there is no opts_ directories at all
            old = - 1

        if old == 25:
            print('Too many optimization folders in {}'.format(os.getcwd()))
            raise ValueError

        new_dir_name = 'opts_{}'.format(old + 1)
        os.mkdir(new_dir_name)
        os.chdir(new_dir_name)

        #copy the python file that create the optimization
        try:
            from shutil import copy2
            copy2(calling_file,os.getcwd())
        except:
            pass

        #create a file with the script
        with open('script_file.lsf','a') as file:
            file.write(self.base_script.replace(';',';\n'))



if __name__=='__main__':
    import numpy as np
    from lumopt.geometries.polygon import function_defined_Polygon, cross
    from lumopt.optimizers.generic_optimizers import ScipyOptimizers
    from lumopt.figures_of_merit.modematch import ModeMatch
    from lumopt.utilities.load_lumerical_scripts import load_from_lsf
    import os
    import matplotlib.pyplot as plt
    from lumopt import CONFIG

    base_script = load_from_lsf(os.path.join(CONFIG['root'], 'examples/crossing/crossing_base_TE_modematch_2D.lsf'))

    fom = ModeMatch(modeorder=2)
    optimizer = ScipyOptimizers(max_iter=20)
    # optimizer=FixedStepGradientDescent(max_dx=20e-9,max_iter=100)
    bounds = [(0.2e-6, 1e-6)]*10
    geometry = function_defined_Polygon(func=cross, initial_params=np.linspace(0.25e-6, 0.6e-6, 10), eps_out='SiO2 (Glass) - Palik',
                                        eps_in=2.8**2, bounds=bounds, depth=220e-9, edge_precision=5)

    opt = Optimization(base_script=base_script, fom=fom, geometry=geometry, optimizer=optimizer)

    opt.run()
