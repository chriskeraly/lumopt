import lumopt.lumerical_methods.lumerical_scripts as ls
import numpy as np
from lumopt import CONFIG
import sys
from fom import fom
from lumopt.utilities.fields import Fields


import lumapi
mu0=4*np.pi*1e-7
c=2.9979e8

class ModeMatch(fom):

    '''Single wavelength Figure of Merit class which is simply the modematch integral of the fields to a propagation
    eigenmode of the cross-section present where the overlap is calculated. The overlap integral is that of equation (7)
    of `https://doi.org/10.1364/OE.21.021693`

    The mode to match to is calculated in Lumerical during the initialization of the optimization. In order to do so, the name of a
    field monitor present in the base simulation at the location where the mode overlap is to be calculated must be provided, as well
    as the modeorder of the eigenmode to match to, the wavelength of interest, and the desired direction of propagation of the mode

    Parameters
    ----------
    :param monitor_name:
        The name of the field monitor in the Lumerical FDTD Base simulation from which the fields will be used to calculate the mode overlap
    :param wavelength:
        The wavelength of interest for the mode overlap calculation. The user should make sure that the source used in the simulation has a bandwidth
        which covers the wavelength of interest
    :param modeorder:
        The modeorder of the mode to couple to when a ModeSource of the same dimensions as the field monitor is set to 'user defined'
        mode selection and at the wavelength of interest
    :param direction:
        The direction of desired propagation energy
'''

    def __init__(self, monitor_name='fom', wavelength=1550e-9,modeorder=1,direction='Forward'):
        self.monitor_name = monitor_name
        self.wavelengths = [wavelength]
        # self.mode=self.normalize_input_mode(mode) #mode is a fields object
        self.current_fom = None
        self.fields = None
        self.modeorder=modeorder
        self.direction=direction
        self.mode=None
        self.inverted_H_mode=None

    def initialize(self,sim, monitor_name='fom', plot=False):
        self.get_mode(sim, monitor_name,plot=plot)

    def get_mode(self, sim, monitor_name='fom', plot=False):
        '''Extracts the desired mode from the base simulation'''

        # TODO: deal with other mode propagation directions (should be ok but to be verified)
        handle = sim.solver_handle

        modesource_name = monitor_name + '_mode_extract'
        lumapi.evalScript(handle, 'addmode; set("name","{}");'.format(modesource_name))


        ls.copy_properties(handle,monitor_name,modesource_name,['x', 'y', 'z', 'x span', 'y span', 'z span'])
        lumapi.evalScript(handle,"set('wavelength start',{});set('wavelength stop',{});".format(self.wavelengths[0],self.wavelengths[0]))

        lumapi.evalScript(handle,"set('direction','{}');".format(self.direction))
        lumapi.evalScript(handle, "save('modeextract');")
        lumapi.evalScript(handle, 'updatesourcemode({});'.format(self.modeorder))

        mode_fields = ls.get_fields_modesource(handle, modesource_name, get_H=True,direction=self.direction)
        mode_fields.normalize_power(plot=plot)
        self.mode = mode_fields

    def put_monitors(self,simulation):
        ''' Make sure the field monitor is looking at the right wavelength'''

        script="select('{}');" \
               "set('override global monitor settings',1);" \
               "set('frequency points',1);" \
               "set('wavelength center',{});".format(self.monitor_name,self.wavelengths[0])


        lumapi.evalScript(simulation.solver_handle,script)

    def get_fom(self, simulation):
        '''Uploads the fields from a completed forward simulation, and performs the mode overlap integral on them'''
        fields = ls.get_fields(simulation.solver_handle, self.monitor_name, get_H=True)
        source_power = np.zeros(np.shape(fields.wl))
        for i, wl in enumerate(fields.wl):
            source_power[i] = ls.get_source_power(simulation.solver_handle, wl=wl)

        self.fields = fields
        self.source_power=source_power
        fom_v_wavelength,phase_preactors = self.mode.calculate_overlap(fields,remove_H=True)

        fom_v_wavelength =np.array(fom_v_wavelength)/np.array(source_power)

        self.phase_prefactors=np.array(phase_preactors)/np.array(source_power)

        # TODO This does not properly deal with multiple wavelengths right now

        return fom_v_wavelength[0]

    def add_adjoint_sources(self, sim):

        pp = self.phase_prefactors[0]
        omega=2*np.pi*c/self.wavelengths[0]
        adjoint_injection_mode=Fields(self.mode.x,self.mode.y,self.mode.z,self.mode.wl,self.mode.E*np.conj(pp)*omega*1j*0,self.mode.D,self.mode.eps,self.mode.H*np.conj(pp)*omega*1j)

          # TODO: This does not deal with multiple wavelengths

        if self.direction=='Forward':
            adjoint_injection_direction='Backward'
        elif self.direction=='Backward':
            adjoint_injection_direction='Forward'
        else:
            raise ValueError('Direction must be Forward or Backward')

        ls.add_imported_source(sim.solver_handle,adjoint_injection_mode,fommonitorname=self.monitor_name,wavelengths=self.wavelengths,direction=adjoint_injection_direction)

        return




if __name__ == '__main__':

    import numpy as np
    from lumopt.optimization import Optimization
    from lumopt.optimizers.generic_optimizers import ScipyOptimizers,FixedStepGradientDescent
    from lumopt.utilities.load_lumerical_scripts import load_from_lsf
    import os
    from lumopt.geometries.polygon import function_defined_Polygon,taper_splitter
    from lumopt.utilities.materials import Material
    from lumopt import CONFIG

    script = load_from_lsf(os.path.join(CONFIG['root'], 'examples/splitter01/splitter_base_TE_modematch.lsf'))



    fom = ModeMatch(modeorder=3)
    #optimizer = ScipyOptimizers(max_iter=20)
    optimizer=FixedStepGradientDescent(max_dx=20e-9,max_iter=20)
    bounds = [(0.2e-6, 1e-6)]*18
    geometry = function_defined_Polygon(func=taper_splitter, initial_params=np.linspace(0.25e-6, 0.6e-6, 18),
                                        eps_out=Material(1.44 ** 2), eps_in=Material(2.8 ** 2, 2), bounds=bounds,
                                        depth=220e-9,
                                        edge_precision=5)

    # geometry=Polygon(eps_in=2.8**2,eps_out=1.44**2)
    opt = Optimization(base_script=script, fom=fom, geometry=geometry, optimizer=optimizer)
    opt.run()
    # opt.initialize()
    #
    #
    # foms = []
    # fom = opt.run_forward_solves()
    # foms.append(fom)
    # # calculate the gradient using the adjoint solve
    # opt.run_adjoint_solves()
    # calculated_gradients = np.array(opt.calculate_gradients())
    # print calculated_gradients
    # n_derivatives = 4
    # finite_difference_gradients = opt.calculate_finite_differences_gradients(n_derivatives=n_derivatives, dx=5e-9,
    #                                                                          central=True, print_res=True)
    #
    # print finite_difference_gradients