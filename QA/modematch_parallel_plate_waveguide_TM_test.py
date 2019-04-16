""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import sys
sys.path.append(".")
import os
import numpy as np

from qatools import *

from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.utilities.wavelengths import Wavelengths
from lumopt.utilities.simulation import Simulation
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimization import Optimization

class TestModeMatchParallelPlateWaveguideTM(TestCase):
    """ 
        Unit test for the ModeMatch class: it performs a quick check that the figure of merit is computed correctly
        using a simple a parallel plate waveguide partially filled by a dielectric. The waveguide has a material interface
        in the middle, and the figure of merit should be the same regardless of the material in which the source is placed.
        This is used to verify that the ModeMatch inputs monitor_name, direction and mode number work correctly.
    """

    file_dir = os.path.abspath(os.path.dirname(__file__))

    def setUp(self):
        # base script
        self.base_script = load_from_lsf(os.path.join(self.file_dir, 'modematch_parallel_plate_waveguide_TM_base.lsf'))
        # bandwidth        
        self.wavelengths = Wavelengths(start = 1540e-9, stop = 1560e-9, points = 3)
        # simulation
        self.sim = Simulation(workingDir = self.file_dir, hide_fdtd_cad = True)
        self.sim.fdtd.eval(self.base_script)
        Optimization.set_global_wavelength(self.sim, self.wavelengths)
        # reference
        self.ref_fom = 0.6643986

    def test_forward_injection_in_3D(self):
        """ Test forward injection in 3D with mode source in vacuum. """
        self.fom = ModeMatch(monitor_name = 'figure_of_merit',
                             mode_number = 1,
                             direction = 'Forward',
                             multi_freq_src = True,
                             target_T_fwd = lambda wl: np.ones(wl.size),
                             norm_p = 1)
        Optimization.set_source_wavelength(self.sim, 'source', self.fom.multi_freq_src, len(self.wavelengths))
        self.sim.fdtd.setnamed('FDTD','dimension','3D')
        self.fom.add_to_sim(self.sim)
        self.sim.run(name = 'modematch_forward_injection_in_3D', iter = 0)
        FOM = self.fom.get_fom(self.sim)
        self.assertAlmostEqual(FOM, self.ref_fom, 5)

    def test_backward_injection_in_3D(self):
        """ Test backward injection in 3D with mode source in dielectric region. """
        self.fom = ModeMatch(monitor_name = 'figure_of_merit',
                             mode_number = 1,
                             direction = 'Backward',
                             multi_freq_src = True,
                             target_T_fwd = lambda wl: np.ones(wl.size),
                             norm_p = 1)
        Optimization.set_source_wavelength(self.sim, 'source', self.fom.multi_freq_src, len(self.wavelengths))
        self.sim.fdtd.setnamed('FDTD','dimension','3D')
        self.sim.fdtd.setnamed('source', 'x', -self.sim.fdtd.getnamed('source','x'))
        self.sim.fdtd.setnamed('source','direction','Backward')
        self.sim.fdtd.setnamed('figure_of_merit','x', -self.sim.fdtd.getnamed('figure_of_merit','x'))
        self.fom.add_to_sim(self.sim)
        self.sim.run(name = 'modematch_backward_injection_in_3D', iter = 1)
        FOM = self.fom.get_fom(self.sim)
        self.assertAlmostEqual(FOM, self.ref_fom, 5)

    def test_forward_injection_in_2D(self):
        """ Test forward injection in 2D with mode source in vacuum. """
        self.fom = ModeMatch(monitor_name = 'figure_of_merit',
                             mode_number = 1,
                             direction = 'Forward',
                             multi_freq_src = True,
                             target_T_fwd = lambda wl: np.ones(wl.size),
                             norm_p = 1)
        Optimization.set_source_wavelength(self.sim, 'source', self.fom.multi_freq_src, len(self.wavelengths))
        self.sim.fdtd.setnamed('FDTD','dimension','2D')
        self.fom.add_to_sim(self.sim)
        self.sim.run(name = 'modematch_forward_injection_in_2D', iter = 2)
        FOM = self.fom.get_fom(self.sim)
        self.assertAlmostEqual(FOM, self.ref_fom, 5)

    def test_no_forward_injection_in_2D(self):
        """ Test no forward injection in 2D with mode source in vacuum. """
        self.fom = ModeMatch(monitor_name = 'figure_of_merit',
                             mode_number = 2, # evanescent mode
                             direction = 'Forward',
                             multi_freq_src = False,
                             target_T_fwd = lambda wl: np.ones(wl.size),
                             norm_p = 1)
        Optimization.set_source_wavelength(self.sim, 'source', self.fom.multi_freq_src, len(self.wavelengths))
        self.sim.fdtd.setnamed('FDTD','dimension','2D')
        self.fom.add_to_sim(self.sim)
        self.sim.run(name = 'modematch_no_forward_injection_in_2D', iter = 3)
        FOM = self.fom.get_fom(self.sim)
        self.assertAlmostEqual(FOM, 0.0, 5)
    
    def test_backward_injection_in_2D(self):
        """ Test backward injection in 2D with mode source in dielectric region. """
        self.fom = ModeMatch(monitor_name = 'figure_of_merit',
                             mode_number = 1,
                             direction = 'Backward',
                             multi_freq_src = True,
                             target_T_fwd = lambda wl: np.ones(wl.size),
                             norm_p = 1)
        Optimization.set_source_wavelength(self.sim, 'source', self.fom.multi_freq_src, len(self.wavelengths))
        self.sim.fdtd.setnamed('FDTD','dimension','2D')
        self.sim.fdtd.setnamed('source', 'x', -self.sim.fdtd.getnamed('source','x'))
        self.sim.fdtd.setnamed('source','direction','Backward')
        self.sim.fdtd.setnamed('figure_of_merit','x', -self.sim.fdtd.getnamed('figure_of_merit','x'))
        self.fom.add_to_sim(self.sim)
        self.sim.run(name = 'modematch_backward_injection_in_2D', iter = 4)
        FOM = self.fom.get_fom(self.sim)
        self.assertAlmostEqual(FOM, self.ref_fom, 5)

if __name__ == "__main__":
    run([__file__])
