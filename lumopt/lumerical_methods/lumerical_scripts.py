""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import numpy as np
import scipy as sp
import scipy.constants
from lumopt.utilities.fields import Fields, FieldsNoInterp


def get_fields_no_interp(fdtd, monitor_name, get_eps, get_D, get_H):
    '''This function fetches fields from a monitor.
    For the fields used to create gradients, the D component of the field and the refractive index also need to be fetched.
    For the fields on figure of merit monitors, D  or index does not need to be fetched
    It returns a Field object'''

    index_monitor_name = monitor_name + '_index'
    
    if get_eps:
        index_x = fdtd.getdata(index_monitor_name, 'index_x')
        index_y = fdtd.getdata(index_monitor_name, 'index_y')
        index_z = fdtd.getdata(index_monitor_name, 'index_z')
        fields_eps_x = np.power(index_x, 2)
        fields_eps_y = np.power(index_y, 2)
        fields_eps_z = np.power(index_z, 2)
        fields_eps = np.stack((fields_eps_x, fields_eps_y, fields_eps_z), axis = -1)
    else:
        fields_eps = None

    fields_E_dict = fdtd.getresult(monitor_name, 'E')
    fields_E = fields_E_dict['E']
    fields_D = fields_E * fields_eps * sp.constants.epsilon_0 if get_D else None

    if get_H:
        fields_H_dict = fdtd.getresult(monitor_name,'H')
        fields_H = fields_H_dict['H']
    else:
        fields_H = None

    monitor_dimension = fdtd.getresult(monitor_name, 'dimension')
    delta_x = fdtd.getresult(monitor_name, 'delta_x')
    delta_y = fdtd.getresult(monitor_name, 'delta_y')
    delta_z = np.array(0.0) if monitor_dimension == 2.0 else fdtd.getresult(monitor_name, 'delta_z')
    deltas = [delta_x, delta_y, delta_z]

    return FieldsNoInterp(fields_E_dict['x'], fields_E_dict['y'], fields_E_dict['z'], fields_E_dict['lambda'], deltas, fields_E ,fields_D, fields_eps, fields_H)

def get_fields_interp(fdtd, monitor_name, get_eps=False, get_D=False, get_H=False):
    '''This function fetches fields from a monitor.
    For the fields used to create gradients, the D component of the field and the refractive index also need to be fetched.
    For the fields on figure of merit monitors, D  or index does not need to be fetched
    It returns a Field object'''

    fields = fdtd.getresult(monitor_name,'E')
    fields_eps = fdtd.getresult(monitor_name + '_D_index','eps')['eps'] if get_eps else None
    fields_D = fdtd.getresult(monitor_name + '_D_index','D')['D'] if get_D else None
    fields_H = fdtd.getresult(monitor_name,'H')['H'] if get_H else None

    return Fields(fields['x'], fields['y'], fields['z'], fields['lambda'], fields['E'], fields_D, fields_eps, fields_H)

def get_fields(fdtd, monitor_name, get_eps, get_D, get_H, nointerpolation):
    '''This function fetches fields from a monitor.
    For the fields used to create gradients, the D component of the field and the refractive index also need to be fetched.
    For the fields on figure of merit monitors, D  or index does not need to be fetched
    It returns a Field object'''

    if nointerpolation:
        return get_fields_no_interp(fdtd, monitor_name, get_eps, get_D, get_H)
    else:
        return get_fields_interp(fdtd, monitor_name, get_eps, get_D, get_H)
