""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import numpy as np
import scipy as sp
import scipy.constants
import lumapi
from lumopt.utilities.fields import Fields, FieldsNoInterp

def get_fields_on_cad(fdtd, monitor_name, field_result_name, get_eps, get_D, get_H, nointerpolation, unfold_symmetry = True):
    unfold_symmetry_string = "true" if unfold_symmetry else "false"
    fdtd.eval("options=struct; options.unfold={0};".format(unfold_symmetry_string) + 
              "{0} = struct;".format(field_result_name) +
              "{0}.E = getresult('{1}','E',options);".format(field_result_name, monitor_name))

    if get_eps or get_D:
        index_monitor_name = monitor_name + '_index'
        fdtd.eval("{0}.index = getresult('{1}','index',options);".format(field_result_name, index_monitor_name))

    if get_H:
        fdtd.eval("{0}.H = getresult('{1}','H',options);".format(field_result_name, monitor_name))

    if nointerpolation:
        fdtd.eval("{0}.delta = struct;".format(field_result_name) +
                  "{0}.delta.x = getresult('{1}','delta_x',options);".format(field_result_name, monitor_name) +
                  "{0}.delta.y = getresult('{1}','delta_y',options);".format(field_result_name, monitor_name))
        monitor_dimension = fdtd.getresult(monitor_name, 'dimension')
        if monitor_dimension == 3:
            fdtd.eval("{0}.delta.z = getdata('{1}','delta_z');".format(field_result_name, monitor_name))
        else:
            fdtd.eval("{0}.delta.z = 0.0;".format(field_result_name))


def get_fields(fdtd, monitor_name, field_result_name, get_eps, get_D, get_H, nointerpolation, unfold_symmetry = True, on_cad_only = False):

    get_fields_on_cad(fdtd, monitor_name, field_result_name, get_eps, get_D, get_H, nointerpolation, unfold_symmetry)

    ## If required, we now transfer the field data to Python and package it up 
    if not on_cad_only:
        fields_dict = lumapi.getVar(fdtd.handle, field_result_name)

    if get_eps:
        if fdtd.getnamednumber('varFDTD') == 1:
            if 'index_x' in fields_dict['index'] and 'index_y' in fields_dict['index'] and not 'index_z' in fields_dict['index']: # varFDTD TE simulation
                fields_dict['index']['index_z'] = fields_dict['index']['index_x']*0.0 + 1.0
            elif not 'index_x' in fields_dict['index'] and not 'index_y' in fields_dict['index'] and 'index_z' in fields_dict['index']: # varFDTD TM simulation
                fields_dict['index']['index_x'] = fields_dict['index']['index_z']*0.0 + 1.0
                fields_dict['index']['index_y'] = fields_dict['index']['index_x']
        assert 'index_x' in fields_dict['index'] and 'index_y' in fields_dict['index'] and 'index_z' in fields_dict['index']
        fields_eps = np.stack((np.power(fields_dict['index']['index_x'], 2), 
                               np.power(fields_dict['index']['index_y'], 2), 
                               np.power(fields_dict['index']['index_z'], 2)), 
                               axis = -1)
    else:
        fields_eps = None

    fields_D = fields_dict['E']['E'] * fields_eps * sp.constants.epsilon_0 if get_D else None

    fields_H = fields_dict['H']['H'] if get_H else None

    if nointerpolation:
        deltas = [fields_dict['delta']['x'], fields_dict['delta']['y'], fields_dict['delta']['z']]
        return FieldsNoInterp(fields_dict['E']['x'], fields_dict['E']['y'], fields_dict['E']['z'], fields_dict['E']['lambda'], deltas, fields_dict['E']['E'], fields_D, fields_eps, fields_H)
    else:
        return Fields(fields_dict['E']['x'], fields_dict['E']['y'], fields_dict['E']['z'], fields_dict['E']['lambda'], fields_dict['E']['E'], fields_D, fields_eps, fields_H)

def set_spatial_interp(fdtd,monitor_name,setting):
    script='select("{}");set("spatial interpolation","{}");'.format(monitor_name,setting)
    fdtd.eval(script)

def get_eps_from_sim(fdtd, monitor_name = 'opt_fields', unfold_symmetry = True):
    index_monitor_name = monitor_name + '_index'

    unfold_symmetry_string = "true" if unfold_symmetry else "false"
    fdtd.eval(('options=struct; options.unfold={0};'
               '{1}_result = getresult("{1}","index",options);'
               '{1}_eps_x = ({1}_result.index_x)^2;'
               '{1}_eps_y = ({1}_result.index_y)^2;'
               '{1}_eps_z = ({1}_result.index_z)^2;'
               '{1}_x = {1}_result.x;'
               '{1}_y = {1}_result.y;'
               '{1}_z = {1}_result.z;'
               '{1}_lambda = {1}_result.lambda;'
               ).format(unfold_symmetry_string, index_monitor_name))
    fields_eps_x = fdtd.getv('{0}_eps_x'.format(index_monitor_name))    # np.power(index_dict['index_x'], 2)
    fields_eps_y = fdtd.getv('{0}_eps_y'.format(index_monitor_name))    # np.power(index_dict['index_y'], 2)
    fields_eps_z = fdtd.getv('{0}_eps_z'.format(index_monitor_name))    # np.power(index_dict['index_z'], 2)
    index_monitor_x      = fdtd.getv('{0}_x'.format(index_monitor_name))  # index_dict['x']
    index_monitor_y      = fdtd.getv('{0}_y'.format(index_monitor_name))   # index_dict['y']
    index_monitor_z      = fdtd.getv('{0}_z'.format(index_monitor_name))  # index_dict['z']
    index_monitor_lambda = fdtd.getv('{0}_lambda'.format(index_monitor_name))  # index_dict['lambda']


    # index_dict = fdtd.getresult(index_monitor_name, 'index')   #< Currently does not work with unfolding options
    # fields_eps_x = np.power(index_dict['index_x'], 2)
    # fields_eps_y = np.power(index_dict['index_y'], 2)
    # fields_eps_z = np.power(index_dict['index_z'], 2)
    # index_monitor_x      = index_dict['x']
    # index_monitor_y      = index_dict['y']
    # index_monitor_z      = index_dict['z']
    # index_monitor_lambda = index_dict['lambda']


    fields_eps = np.stack((fields_eps_x, fields_eps_y, fields_eps_z), axis = -1)
    return fields_eps, index_monitor_x,index_monitor_y,index_monitor_z, index_monitor_lambda
