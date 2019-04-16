""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import os
from inspect import signature

from lumapi import FDTD
from lumopt.utilities.load_lumerical_scripts import load_from_lsf

class BaseScript(object):
    """ 
        Proxy class for creating a base simulation. It acts as an interface to place the appropriate call in the FDTD CAD
        to build the base simulation depending on the input object. Options are:
            1) a Python callable,
            2) any visible *.fsp project file,
            3) any visible *.lsf script file or
            4) a plain string with a Lumerical script.
        
        Parameters:
        -----------
        :script_obj: executable, file name or plain string.
    """

    def __init__(self, script_obj):
        if callable(script_obj):
            self.callable_obj = script_obj
            params = signature(script_obj).parameters
            if len(params) > 1:
                raise UserWarning('function to create base simulation must take a single argument (handle to FDTD CAD).')
        elif isinstance(script_obj, str):
            if '.fsp' in script_obj and os.path.isfile(script_obj):
                self.project_file = os.path.abspath(script_obj)
            elif '.lsf' in script_obj and os.path.isfile(script_obj):
                self.script_str = load_from_lsf(os.path.abspath(script_obj))
            else:
                self.script_str = str(script_obj)
        else:
            raise UserWarning('object for generating base simulation must be a Python function, a file name or a string with a Lumerical script.')

    def __call__(self, cad_handle):
        return self.eval(cad_handle)

    def eval(self, cad_handle):
        if not isinstance(cad_handle, FDTD):
            raise UserWarning('input must be handle returned by lumapi.FDTD.')
        if hasattr(self, 'callable_obj'):
            return self.callable_obj(cad_handle)
        elif hasattr(self, 'project_file'):
            return cad_handle.load(self.project_file)
        elif hasattr(self, 'script_str'):
            return cad_handle.eval(self.script_str)
        else:
            raise RuntimeError('un-initialized object.')