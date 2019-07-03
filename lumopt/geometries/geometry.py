""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import sys
import numpy as np
import lumapi

class Geometry(object):

    self_update=False
    unfold_symmetry = True #< By default, we do want monitors to unfold symmetry

    def use_interpolation(self):
        return False
        
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

    def update_geometry(self, params, sim = None):
        if self.operation=='mul':
            for geometry in self.geometries:
                geometry.update_geometry(params,sim)

        if self.operation=='add':
            n1=len(self.geometries[0].get_current_params())
            self.geometries[0].update_geometry(params[:n1],sim)
            self.geometries[1].update_geometry(params[n1:],sim)

    def calculate_gradients(self, gradient_fields):
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

    def add_geo(self, sim, params, only_update):
        for geometry in self.geometries:
            geometry.add_geo(sim, params, only_update)

    @staticmethod
    def get_eps_from_index_monitor(fdtd, eps_result_name, monitor_name = 'opt_fields'):
        index_monitor_name = monitor_name + '_index'
        fdtd.eval("{0}_data_set = getresult('{0}','index');".format(index_monitor_name) +
                  "{0} = matrix(length({1}_data_set.x), length({1}_data_set.y), length({1}_data_set.z), length({1}_data_set.f), 3);".format(eps_result_name, index_monitor_name) +
                  "{0}(:, :, :, :, 1) = {1}_data_set.index_x^2;".format(eps_result_name, index_monitor_name) +
                  "{0}(:, :, :, :, 2) = {1}_data_set.index_y^2;".format(eps_result_name, index_monitor_name) +
                  "{0}(:, :, :, :, 3) = {1}_data_set.index_z^2;".format(eps_result_name, index_monitor_name) +
                  "clear({0}_data_set);".format(index_monitor_name))

    def d_eps_on_cad(self, sim):
        Geometry.get_eps_from_index_monitor(sim.fdtd, 'original_eps_data')
        current_params = self.get_current_params()
        sim.fdtd.eval("d_epses = cell({});".format(current_params.size))
        lumapi.putDouble(sim.fdtd.handle, "dx", self.dx)
        print('Getting d eps: dx = ' + str(self.dx))
        sim.fdtd.redrawoff()
        for i,param in enumerate(current_params):
            d_params = current_params.copy()
            d_params[i] = param + self.dx
            self.add_geo(sim, d_params, only_update = True)
            Geometry.get_eps_from_index_monitor(sim.fdtd, 'current_eps_data')
            sim.fdtd.eval("d_epses{"+str(i+1)+"} = (current_eps_data - original_eps_data) / dx;")
            sys.stdout.write('.'), sys.stdout.flush()
        sim.fdtd.eval("clear(original_eps_data, current_eps_data, dx);")
        print('')
        sim.fdtd.redrawon()
