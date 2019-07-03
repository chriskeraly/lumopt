""" Copyright (c) 2019 Lumerical Inc. """

import sys
sys.path.append(".")
import os

from qatools import *

from lumopt.utilities.simulation import Simulation
from lumopt.utilities.base_script import BaseScript

class TestVarFDTDBaseScript(TestCase):
    """ 
        Unit test for BaseScript class. It verifies that the object is able to run an *.lsf script, a *.lms project file or a plain script in a string.
    """

    file_dir = os.path.abspath(os.path.dirname(__file__))

    def setUp(self):
        self.sim = Simulation(workingDir = self.file_dir, use_var_fdtd = True, hide_fdtd_cad = True)

    def test_eval_project_file(self):
        my_project_file = os.path.join(self.file_dir,'base_varfdtd_script_test.lms')
        base_script_obj = BaseScript(my_project_file)
        base_script_obj(self.sim.fdtd)
        self.assertTrue(self.sim.fdtd.getnamednumber('varFDTD') == 1)
        self.assertTrue(self.sim.fdtd.getnamednumber('polygon') == 1)

    def test_eval_python_script(self):
        my_fun = lambda fdtd_handle: fdtd_handle.addvarfdtd()
        base_script_obj = BaseScript(my_fun)
        base_script_obj.eval(self.sim.fdtd)
        self.assertTrue(self.sim.fdtd.getnamednumber('varFDTD') == 1)

    def test_eval_script_file(self):
        my_script_file = os.path.join(self.file_dir,'base_varfdtd_script_test.lsf')
        base_script_obj = BaseScript(my_script_file)
        base_script_obj.eval(self.sim.fdtd)
        self.assertTrue(self.sim.fdtd.getnamednumber('varFDTD') == 1)

    def test_eval_script(self):
        my_script = "load('base_varfdtd_script_test.fsp');"
        base_script_obj = BaseScript(my_script)
        base_script_obj.eval(self.sim.fdtd)
        self.assertTrue(self.sim.fdtd.getnamednumber('varFDTD') == 1)
        self.assertTrue(self.sim.fdtd.getnamednumber('polygon') == 1)

if __name__ == "__main__":
    run([__file__])
