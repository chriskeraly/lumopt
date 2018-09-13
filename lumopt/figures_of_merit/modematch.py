import lumopt.lumerical_methods.lumerical_scripts as ls
from numpy import conj, pi
import numpy as np
from lumopt import CONFIG
import sys
from fom import fom



import lumapi
mu0=4*np.pi*1e-7
c=2.9979e8

class ModeMatch(fom):
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
        self.direction=direction

    def get_mode(self, sim,  monitor_name='fom', source_name='source', plot=False):
        # TODO: deal with other mode propagation directions
        handle = sim.solver_handle

        modesource_name = monitor_name + '_mode_extract'
        lumapi.evalScript(handle, 'addmode; set("name","{}");'.format(modesource_name))
        ls.copy_properties(handle,monitor_name,modesource_name,['x', 'y', 'z', 'x span', 'y span', 'z span'])
        lumapi.evalScript(handle, "set('wavelength start',{});set('wavelength stop',{});".format(self.wavelengths[0],
                                                                                                 self.wavelengths[0]))
        ls.set_injection_axis(handle,modesource_name)
        lumapi.evalScript(handle,"set('direction','{}');".format(self.direction))
        ls.copy_properties(handle, monitor_name, modesource_name, ['x', 'y', 'z', 'x span', 'y span', 'z span'])
        lumapi.evalScript(handle, "save('modeextract');")
        lumapi.evalScript(handle, 'updatesourcemode({});'.format(self.modeorder))

        mode_fields = ls.get_fields_modesource(handle, modesource_name, get_H=True,direction=self.direction)
        mode_fields.normalize_power(plot=plot)
        self.mode = mode_fields


    def initialize(self,sim, monitor_name='fom', plot=False):
        self.get_mode(sim, monitor_name,source_name='source',plot=plot)
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
                    if normal==[1,0,0]:
                        phase_factor = np.conj([0,Hm[2],-Hm[1]])/2 #for mode everlap calculation, and because the field is only being calculated in one direction
                    elif normal==[0,1,0]:
                        phase_factor = np.conj([-Hm[2],0,Hm[0]])/2 #for mode everlap calculation, and because the field is only being calculated in one direction
                    elif normal == [0, 0, 1]:
                        phase_factor = np.conj([Hm[1],-Hm[0],0])/2  # for mode everlap calculation, and because the field is only being calculated in one direction

                    phase_factors.append(phase_factor*dy*dx*dz)

        self.positions=positions
        self.phase_factors=phase_factors

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


    def add_adjoint_sources(self, simulation):
        fields = self.fields
        pointfields = [fields.getfield(pos[0], pos[1], pos[2], self.wavelength) for pos in self.positions]

        sum_of_pointfields = sum(
            [pointfield*phase_factor for pointfield, phase_factor in zip(pointfields, self.phase_factors)])
        prefactor = np.conj(sum_of_pointfields)
        adjoint_sources = [prefactor*phase_factor for phase_factor in self.phase_factors]

        adjoint_sources = adjoint_sources/self.source_power
        script = ''
        for i, (adjoint_source, pos) in enumerate(zip(adjoint_sources, self.positions)):
            script += ls.add_dipole_script(simulation.solver_handle, pos[0], pos[1], pos[2], self.wavelength,
                                           adjoint_source, name_suffix=str(i))
        ls.lumapi.evalScript(simulation.solver_handle, script)
        return

    def put_monitors(self,simulation):
        pass

        return

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