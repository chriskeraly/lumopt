import lumopt.lumerical_methods.lumerical_scripts as ls
from fom import fom
import numpy as np
import lumapi

class FieldIntensity(fom):
    '''

    A figure of merit which is simply |E|^2 at a point monitor defined in the base simulation

    '''

    def __init__(self, monitor_name='fom', wavelengths=[1550e-9],long_source=False):
        '''
        :param monitor_name: A string: the name of the point monitor
        :param wavelengths: A list of the wavelengths of interest (for the moment supports only a single value)
        '''
        self.monitor_name = monitor_name
        self.wavelengths = wavelengths
        self.current_fom = None
        self.fields = None
        self.long_source=long_source


    def get_fom(self, simulation):
        '''

        :param simulation: The simulation object of the base simulation
        :return: The figure of merit
        '''
        field = ls.get_fields(simulation.solver_handle, self.monitor_name)
        self.fields = field

        pointfield = field.getfield(field.x[0], field.y[0], field.z[0], self.wavelengths[0])
        fom = sum(pointfield*np.conj(pointfield))

        return fom

    def add_adjoint_sources(self, sim):
        '''
        Adds the adjoint sources required in the adjoint simulation

        :param simulation: The simulation object of the base simulation
        '''

        field = self.fields
        pointfield = field.getfield(field.x[0], field.y[0], field.z[0], self.wavelengths[0])

        # print prefactor
        adjoint_source = np.conj(pointfield)

        ls.add_dipole(sim.fdtd, field.x[0], field.y[0], field.z[0], self.wavelengths[0], adjoint_source)
        if self.long_source:
            script='set("optimize for short pulse", 0);'
            sim.fdtd.eval(script)


class FieldIntensities(fom):
    '''
    A slightly more complex figure of merit than FieldIntensity. Now fields at different points in space can be combined linearly
    to create a figure of merit of the form:

    FOM= |a1*E1+ a2*E2 + ... + an*En|^2

    where ai can be a complex vector of length 3.

    With the right amplitudes ai, this can be used to form a wide variety of complex meaningful figures of merit (absorption, modematching)

    If the amplitudes are set to None, then
    FOM=|E1|^2+...+|En|^2
    '''
    
    def __init__(self, positions=[(1.25e-6, 0.36e-6, 0), (1.25e-6, -0.36e-6, 0)], weight_amplitudes=[1, 1], wavelengths=[1550e-9], normalize_to_source_power=False):
        '''

        :param positions: A list of tuples representing the 3D coordinate in space of where the fields should be measured
        :param weight_amplitudes: A list of complex amplitudes
        :param wavelengths: The wavelengths of interest (for the moment supports only one wavelength)
        :param normalize_to_source_power: Should everything be normalized to the source power?
        '''
        self.positions=positions
        self.weight_amplitudes=weight_amplitudes
        self.wavelengths=wavelengths
        self.current_fom=None
        self.fields=None
        self.normalize_to_source_power=normalize_to_source_power
        self.monitor_names=['fom_mon_{}'.format(i) for i in range(len(self.positions))]

    def put_monitors(self,simulation):
        script=''
        for monitor_name,position in zip(self.monitor_names,self.positions):
            script+=ls.add_point_monitor_script(monitor_name,position)
        sim.fdtd.eval(script)

        
    def get_fom(self,simulation):

        fields=[ls.get_fields(simulation.solver_handle, monitor_name) for monitor_name in self.monitor_names]

        if self.normalize_to_source_power:
            source_power = np.zeros(np.shape(fields[0].wl))
            for i, wl in enumerate(fields[0].wl):
                source_power[i] = ls.get_source_power(simulation.solver_handle, wl=wl)
            self.source_power = source_power
        self.fields=fields

        pointfields=[field.getfield(field.x[0],field.y[0],field.z[0],self.wavelengths[0]) for field in fields]
        if self.weight_amplitudes is None:
            fom=sum([sum(pointfield*np.conj(pointfield)) for pointfield in pointfields])
        else:
            sum_of_pointfields=sum([pointfield*phase_factor for pointfield,phase_factor in zip(pointfields, self.weight_amplitudes)])
            fom=sum(sum_of_pointfields*np.conj(sum_of_pointfields))
            if self.normalize_to_source_power:
                fom=fom/np.array(source_power)
        return fom


    def add_adjoint_sources(self, sim):

        fields=self.fields
        pointfields=[field.getfield(field.x[0],field.y[0],field.z[0],self.wavelengths[0]) for field in fields]


        prefactor=1#eps0#*omega
        #print prefactor
        if self.weight_amplitudes is None:
            adjoint_sources=[prefactor*np.conj(pointfield) for pointfield in pointfields]
        else:
            pointfields = [field.getfield(field.x[0], field.y[0], field.z[0], self.wavelengths[0]) for field in fields]
            sum_of_pointfields=sum([pointfield*phase_factor for pointfield,phase_factor in zip(pointfields, self.weight_amplitudes)])
            prefactor=np.conj(sum_of_pointfields)
            adjoint_sources=[prefactor*phase_factor for phase_factor in self.weight_amplitudes]

        if self.normalize_to_source_power:
            adjoint_sources=adjoint_sources/self.source_power
        script=''
        for i,(adjoint_source,field) in enumerate(zip(adjoint_sources,fields)):
            script+=ls.add_dipole_script(field.x[0],field.y[0],field.z[0],self.wavelengths[0],adjoint_source,name_suffix=str(i))
        sim.fdtd.eval(script)
        return

if __name__=='__main__':

    import numpy as np
    from lumopt.forward_problem import Forward_problem
    from lumopt.optimization import Optimization
    from lumopt.optimizers.generic_optimizers import FixedStepGradientDescent
    from lumopt.geometries.polygon import Polygon, Polygon_constrained

    from lumopt.utilities.load_lumerical_scripts import load_from_lsf
    import os
    import matplotlib.pyplot as plt
    from lumopt import CONFIG

    script = load_from_lsf(os.path.join(CONFIG['root'], 'examples/splitter01/splitter_base_TE_nomonitors_3D.lsf'))

    forward_problem = Forward_problem(script=script)

    fom = FieldIntensities([(1.25e-6, 0.36e-6, 0), (1.25e-6, -0.36e-6, 0)], weight_amplitudes=[1, 1j])
    optimizer = FixedStepGradientDescent(dx=5e-9, max_iter=25)

    points_up = zip(np.linspace(-1, 1, 20)*1e-6, np.linspace(0.25, 0.6, 20)*1e-6)
    points_down = zip(np.linspace(-1, 1, 20)*1e-6, np.linspace(-0.25, -0.6, 20)*1e-6)
    points = np.array(points_up[::-1] + points_down)

    linear_points = np.reshape(points, (-1))
    mask = [1 if i%2 == 1 else 0 for i, elem in enumerate(linear_points)]  # this should blockoff all x coordinates
    mask[1] = 0  # first corner
    mask[19*2 + 1] = 0
    mask[20*2 + 1] = 0
    mask[-1] = 0
    bounds = [(0, 1e-6)]*18 + [(-1e-6, 0)]*18

    geometry = Polygon_constrained(eps_out=1.44 ** 2, eps_in=2.8 ** 2, points=points, depth=220e-9, mask=mask,
                                   bounds=bounds)
    # geometry=Polygon(eps_in=2.8**2,eps_out=1.44**2)
    opt = Optimization(forward_problem=forward_problem, fom=fom, geometry=geometry, optimizer=optimizer,
                       solver='lumerical')

    foms = []
    fom = opt.run_forward_solves()
    foms.append(fom)
    # calculate the gradient using the adjoint solve
    opt.run_adjoint_solves()
    calculated_gradients = np.array(opt.calculate_gradients())
    print calculated_gradients



        
        