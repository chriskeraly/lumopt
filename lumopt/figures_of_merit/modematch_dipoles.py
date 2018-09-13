import lumopt.lumerical_methods.lumerical_scripts as ls
from numpy import conj, pi
import numpy as np
from lumopt import CONFIG
import sys
from lumopt.figures_of_merit.field_intensities import FieldIntensities
from fom import fom
from lumopt.utilities.fields import Fields


import lumapi
mu0=4*np.pi*1e-7
c=2.9979e8

class ModeMatch(fom):


    def __init__(self, monitor_name='fom', wavelength=1550e-9,precision=10,modeorder=1,direction='Forward'):
        self.monitor_name = monitor_name
        self.wavelength = wavelength
        self.wavelengths = [wavelength]
        # self.mode=self.normalize_input_mode(mode) #mode is a fields object
        self.current_fom = None
        self.fields = None
        self.precision=precision #number of dipoles per axis
        self.modeorder=modeorder
        self.mode=None
        self.direction = direction

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


    def get_fom(self, simulation):

        fields = ls.get_fields(simulation.solver_handle, self.monitor_name, get_H=True)
        source_power = np.zeros(np.shape(fields.wl))
        for i, wl in enumerate(fields.wl):
            source_power[i] = ls.get_source_power(simulation.solver_handle, wl=wl)

        self.fields = fields
        self.source_power=source_power
        fom_v_wavelength,phase_preactors = self.mode.calculate_overlap(fields)

        fom_v_wavelength =np.array(fom_v_wavelength)/np.array(source_power)

        self.phase_prefactors=np.array(phase_preactors)/np.array(source_power)

        # TODO This does not properly deal with multiple wavelengths right now

        return fom_v_wavelength[0]

    def add_adjoint_sources(self, simulation):

        # TODO: make this work not just for 2D linear X-normal
        # TODO: Make the figure of merit happen on the actual fdtd mesh, not on an interpolated mesh. If the geometry is also
        # computed on the actual fdtd mesh, with a 2nd order continuous geometry defined on the mesh, this should be
        # just as good as discrete adjoint (I believe)

        if len(self.fields.x)!=1:
            x_dip=np.linspace(min(self.fields.x),max(self.fields.x),self.precision)
            dx = x_dip[1] - x_dip[0]
        else:
            x_dip=np.array(self.fields.x)
            dx=1

        if len(self.fields.y) != 1:
            y_dip = np.linspace(min(self.fields.y), max(self.fields.y), self.precision)
            dy = y_dip[1] - y_dip[0]
        else:
            y_dip = np.array(self.fields.y)
            dy = 1

        if len(self.fields.z) != 1:
            z_dip = np.linspace(min(self.fields.z), max(self.fields.z), self.precision)
            dz = z_dip[1] - z_dip[0]
        else:
            z_dip = np.array(self.fields.z)
            dz = 1

        Hm_interpolator = self.mode.getHfield
        Em_interpolator = self.mode.getfield

        px_dip = []
        py_dip = []
        pz_dip = []
        mx_dip = []
        my_dip = []
        mz_dip = []

        pp=self.phase_prefactors[0] #TODO: This does not deal with multiple wavelengths

        i=0
        script = ''
        for x in x_dip:
            for z in z_dip:
                for y in y_dip:
                    Hm = Hm_interpolator(x, y, z, self.wavelength)
                    Em = Em_interpolator(x, y, z, self.wavelength)
                    px_dip.append(0)
                    py_dip.append(np.conj(Hm[2])*np.conj(pp)*dy*dx*dz)
                    pz_dip.append(-np.conj(Hm[1])*np.conj(pp)*dy*dx*dz)
                    mx_dip.append(0)
                    my_dip.append(np.conj(Em[2])*np.conj(pp)/mu0*dy*dx*dz)
                    mz_dip.append(-np.conj(Em[1])*np.conj(pp)/mu0*dy*dx*dz)

                    script += ls.add_dipole_script(simulation.solver_handle, x, y, z, self.wavelength,
                                                   [px_dip[i], py_dip[i], pz_dip[i]],
                                                   name_suffix='electric_{}_'.format(i))
                    script += ls.add_dipole_script(simulation.solver_handle, x, y, z, self.wavelength,
                                                   [mx_dip[i], my_dip[i], mz_dip[i]], magnetic=True,
                                                   name_suffix='magnetic_{}_'.format(i))
                    i+=1


        lumapi.evalScript(simulation.solver_handle,script)

        return


class Modematch_field_source(ModeMatch):

    def __init__(self, monitor_name='fom', wavelengths=[1550e-9],modeorder=1):
        self.monitor_name = monitor_name
        self.wavelengths = wavelengths
        # self.mode=self.normalize_input_mode(mode) #mode is a fields object
        self.current_fom = None
        self.fields = None
        self.modeorder=modeorder
        self.mode=None
        self.inverted_H_mode=None

    def add_adjoint_sources(self, simulation):

        pp = self.phase_prefactors[0]
        omega=2*np.pi*c/self.wavelengths[0]
        adjoint_injection_mode=Fields(self.mode.x,self.mode.y,self.mode.z,self.mode.wl,self.mode.E*np.conj(pp)*omega*1j,self.mode.D,self.mode.eps,self.mode.H*np.conj(pp)*omega*1j)

          # TODO: This does not deal with multiple wavelengths

        ls.add_imported_source(simulation.solver_handle,adjoint_injection_mode,prefactor=1,fommonitorname=self.monitor_name,wavelengths=self.wavelengths)

        return


class ModeMatch2(fom):
    '''Adds it's own monitors'''

    def __init__(self, monitor_name='fom', wavelength=1550e-9,precision=10,modeorder=1,direction='Forward'):
        self.monitor_name = monitor_name
        self.wavelength = wavelength
        self.wavelengths=[wavelength]
        # self.mode=self.normalize_input_mode(mode) #mode is a fields object
        self.current_fom = None
        self.fields = None
        self.precision=precision #number of dipoles per axis
        self.field_intensities_fom_object=None
        self.modeorder=modeorder
        self.direction = direction


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


    def initialize(self,sim, monitor_name='fom', plot=False):
        self.get_mode(sim, monitor_name,plot=plot)
        if len(self.mode.z) != 1:
            z_span=max(self.mode.z)-min(self.mode.z)
            z_dip = np.linspace(min(self.mode.z)+z_span/1000, max(self.mode.z)-z_span/1000, self.precision)
            dz = z_dip[1] - z_dip[0]
        else:
            z_dip = np.array(self.mode.z)
            dz = 1
            normal = [0,0,1]


        if len(self.mode.x)!=1:
            x_span=max(self.mode.x)-min(self.mode.x)
            x_dip=np.linspace(min(self.mode.x)+x_span/1000,max(self.mode.x)-x_span/1000,self.precision)
            dx = x_dip[1] - x_dip[0]
        else:
            x_dip=np.array(self.mode.x)
            dx=1
            normal = [1,0,0]

        if len(self.mode.y) != 1:
            y_span=max(self.mode.y)-min(self.mode.y)
            y_dip = np.linspace(min(self.mode.y)+y_span/1000, max(self.mode.y)-y_span/1000, self.precision)
            dy = y_dip[1] - y_dip[0]
        else:
            y_dip = np.array(self.mode.y)
            dy = 1
            normal=[0,1,0]

        Hm_interpolator = self.mode.getHfield
        Em_interpolator = self.mode.getfield

        positions=[]
        phase_factors=[]

        for x in x_dip:
            for z in z_dip:
                for y in y_dip:
                    positions.append((x,y,z))

                    Hm = Hm_interpolator(x, y, z, self.wavelength)
                    Em = Em_interpolator(x, y, z, self.wavelength)
                    phase_factor=np.conj([0,Hm[2],-Hm[1]])/2 #for mode overlap calculation, and because the field is only being calculated in one direction
                    phase_factors.append(phase_factor*dy*dx*dz)

        self.field_intensities_fom_object=FieldIntensities(positions=positions, weight_amplitudes=phase_factors, normalize_to_source_power=True)
        self.positions=positions
        self.phase_factors=phase_factors


    def get_fom(self, simulation):
        self.current_fom=self.field_intensities_fom_object.get_fom(simulation)
        return self.current_fom

    def add_adjoint_sources(self, simulation):
        self.field_intensities_fom_object.add_adjoint_sources(simulation)

    def put_monitors(self,simulation):
        self.field_intensities_fom_object.put_monitors(simulation)

        return

class ModeMatch3(ModeMatch2):
    '''Mode matching figure of merit that uses a mode on a regular grid.
    Fetches all the information from one monitor, and puts dipoles as the adjoint sources.
    TODO: add the magnetic dipoles'''


    def get_fom(self, simulation):
        fields = ls.get_fields(simulation.solver_handle, self.monitor_name, get_H=True)
        self.fields=fields
        source_power = np.zeros(np.shape(fields.wl))
        for i, wl in enumerate(fields.wl):
            source_power[i] = ls.get_source_power(simulation.solver_handle, wl=wl)
        self.source_power=source_power

        pointfields=[fields.getfield(pos[0],pos[1],pos[2],self.wavelength) for pos in self.positions]

        sum_of_pointfields = sum([pointfield*phase_factor for pointfield, phase_factor in zip(pointfields, self.phase_factors)])
        fom = sum(sum_of_pointfields*np.conj(sum_of_pointfields))

        fom = fom/np.array(source_power)

        # TODO This does not properly deal with multiple wavelengths right now

        return fom

    def put_monitors(self,simulation):
        pass

    def add_adjoint_sources(self,simulation):

        fields=self.fields
        pointfields=[fields.getfield(pos[0],pos[1],pos[2],self.wavelength) for pos in self.positions]

        prefactor=1#eps0#*omega
        #print prefactor
        sum_of_pointfields=sum([pointfield*phase_factor for pointfield,phase_factor in zip(pointfields,self.phase_factors)])
        prefactor=np.conj(sum_of_pointfields)
        adjoint_sources=[prefactor*phase_factor for phase_factor in self.phase_factors]



        adjoint_sources=adjoint_sources/self.source_power
        script=''
        for i,(adjoint_source,pos) in enumerate(zip(adjoint_sources,self.positions)):
            script+=ls.add_dipole_script(simulation.solver_handle,pos[0],pos[1],pos[2],self.wavelength,adjoint_source,name_suffix=str(i))
        ls.lumapi.evalScript(simulation.solver_handle,script)
        return

class Modematch_combined(ModeMatch3):

    def get_fom(self, simulation):
        '''Uploads the fields from a completed forward simulation, and performs the mode overlap integral on them'''
        fields = ls.get_fields(simulation.solver_handle, self.monitor_name, get_H=True)
        source_power = np.zeros(np.shape(fields.wl))
        for i, wl in enumerate(fields.wl):
            source_power[i] = ls.get_source_power(simulation.solver_handle, wl=wl)

        self.fields = fields
        self.source_power=source_power
        fom_v_wavelength,phase_preactors = self.mode.calculate_overlap(fields)

        fom_v_wavelength =np.array(fom_v_wavelength)/np.array(source_power)

        self.phase_prefactors=np.array(phase_preactors)/np.array(source_power)

        # TODO This does not properly deal with multiple wavelengths right now

        return fom_v_wavelength[0]


if __name__ == '__main__':
    # import matplotlib as mpl
    #
    #  #mpl.use('TkAgg')
    # import matplotlib.pyplot as plt
    # from examples.splitter01.make_sim import make_sim_modematch
    #
    # sim = make_sim_modematch()
    # fom = ModeMatch2()
    # fom.initialize(sim, modeorder=3)
    # fom.mode.plot()
    # plt.show(block=True)
    # print 't'
    # from time import sleep
    #
    # sleep(100)
    import numpy as np
    from lumopt.optimization import Base_Optimization
    from lumopt.optimizers.generic_optimizers import ScipyOptimizers
    from lumopt.utilities.load_lumerical_scripts import load_from_lsf
    import os
    from lumopt.geometries.polygon import function_defined_Polygon,taper_splitter
    from lumopt.utilities.materials import Material
    from lumopt import CONFIG

    script = load_from_lsf(os.path.join(CONFIG['root'], 'examples/splitter01/splitter_base_TE_modematch.lsf'))

    fom = ModeMatch3(modeorder=3,precision=50)
    optimizer = ScipyOptimizers(max_iter=20)

    bounds = [(0.2e-6, 1e-6)]*18
    geometry = function_defined_Polygon(func=taper_splitter, initial_params=np.linspace(0.25e-6, 0.6e-6, 18),
                                        eps_out=Material(1.44 ** 2), eps_in=Material(2.8 ** 2, 2), bounds=bounds, depth=220e-9,
                                        edge_precision=5)


    # geometry=Polygon(eps_in=2.8**2,eps_out=1.44**2)
    opt = Base_Optimization(base_script=script, fom=fom, geometry=geometry, optimizer=optimizer)
    opt.initialize()
    opt.run_forward_solves()
    opt.make_adjoint_solves()

    fom = Modematch_field_source(modeorder=3)
    optimizer = ScipyOptimizers(max_iter=20)

    bounds = [(0.2e-6, 1e-6)]*18
    geometry = function_defined_Polygon(func=taper_splitter, initial_params=np.linspace(0.25e-6, 0.6e-6, 18),
                                        eps_out=Material(1.44 ** 2), eps_in=Material(2.8 ** 2, 2), bounds=bounds,
                                        depth=220e-9,
                                        edge_precision=5)

    # geometry=Polygon(eps_in=2.8**2,eps_out=1.44**2)
    opt = Base_Optimization(base_script=script, fom=fom, geometry=geometry, optimizer=optimizer)
    opt.initialize()
    opt.run_forward_solves()
    opt.make_adjoint_solves()

    from time import sleep
    sleep(36000)
    print 'hello'

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