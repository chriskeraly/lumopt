""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import sys
import numpy as np
import lumapi
from lumopt.utilities.scipy_wrappers import trapz3D

class Geometry(object):

    self_update=False

    def __init__(self,geometries,operation):
        self.geometries=geometries
        self.operation=operation
        if self.operation=='mul':
            self.bounds=geometries[0].bounds
        if self.operation=='add':
            self.bounds=np.concatenate((np.array(geometries[0].bounds),np.array(geometries[1].bounds)))
        self.dx=max([geo.dx for geo in self.geometries])
        return

    def __add__(self,other):
        '''Two geometries with independent parameters'''
        geometries=[self,other]
        return Geometry(geometries,'add')

    def __mul__(self,other):
        '''Two geometries with common parameters'''
        geometries = [self, other]
        return Geometry(geometries, 'mul')

    def add_geo(self, sim, params, only_update):
        for geometry in self.geometries:
            geometry.add_geo(sim, params, only_update)

    def initialize(self,wavelengths,opt):
        for geometry in self.geometries:
            geometry.initialize(wavelengths,opt)
        self.opt=opt

    def update_geometry(self,params):
        if self.operation=='mul':
            for geometry in self.geometries:
                geometry.update_geometry(params)

        if self.operation=='add':
            n1=len(self.geometries[0].get_current_params())
            self.geometries[0].update_geometry(params[:n1])
            self.geometries[1].update_geometry(params[n1:])

    def calculate_gradients(self, gradient_fields, wavelength):
        derivs1 = np.array(self.geometries[0].calculate_gradients(gradient_fields))
        derivs2 = np.array(self.geometries[1].calculate_gradients(gradient_fields))

        if self.operation=='mul':
            return derivs1+derivs2
        if self.operation=='add':
            np.concatenate(derivs1,derivs2)

    def get_current_params(self):
        params1=np.array(self.geometries[0].get_current_params())
        if self.operation=='mul':
            return params1
        if self.operation=='add':
            return params1+np.array(self.geometries[1].get_current_params())

    def plot(self,*args):
        return False

    def update_geo_in_sim(self, sim, params):
        for geometry in self.geometries:
            geometry.update_geo_in_sim(sim, params)

    @staticmethod
    def get_eps_from_sim(fdtd, monitor_name = 'opt_fields'):
        index_monitor_name = monitor_name + '_index'
        index_dict = fdtd.getresult(index_monitor_name, 'index')
        fields_eps_x = np.power(index_dict['index_x'], 2)
        fields_eps_y = np.power(index_dict['index_y'], 2)
        fields_eps_z = np.power(index_dict['index_z'], 2)
        fields_eps = np.stack((fields_eps_x, fields_eps_y, fields_eps_z), axis = -1)
        return fields_eps, index_dict['x'], index_dict['y'], index_dict['z'], index_dict['lambda']

    def get_eps_update(self, sim, params):
        self.update_geo_in_sim(sim, params)
        eps, x, y, z, wl = Geometry.get_eps_from_sim(sim.fdtd)
        return eps

    def get_d_eps(self, sim):
        current_eps, x, y, z, wl = Geometry.get_eps_from_sim(sim.fdtd)
        current_params = self.get_current_params()
        d_epses = list()
        print('Getting d eps: dx = ' + str(self.dx))
        for i,param in enumerate(current_params):
            d_params = current_params.copy()
            d_params[i] = param + self.dx
            d_eps = (self.get_eps_update(sim,d_params) - current_eps) / self.dx
            d_epses.append(d_eps)
            sys.stdout.write('.'), sys.stdout.flush()
        print('')
        return d_epses
