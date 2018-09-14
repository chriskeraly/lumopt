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
        field = ls.get_fields(simulation.fdtd, self.monitor_name)
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
        simulation.fdtd.eval(script)

        
    def get_fom(self,simulation):

        fields=[ls.get_fields(simulation.fdtd, monitor_name) for monitor_name in self.monitor_names]

        if self.normalize_to_source_power:
            source_power = np.zeros(np.shape(fields[0].wl))
            for i, wl in enumerate(fields[0].wl):
                source_power[i] = ls.get_source_power(simulation.fdtd, wl=wl)
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
    pass



        
        