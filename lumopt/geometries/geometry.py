""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import sys
import numpy as np
import lumapi
import lumopt.lumerical_methods.lumerical_scripts as ls
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

    def add_geo(self, sim,params=None):
        for geometry in self.geometries:
            geometry.add_geo(sim,params=params)

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

    def calculate_gradients(self, gradient_fields, wavelength,real=True):
        derivs1 = np.array(self.geometries[0].calculate_gradients(gradient_fields, wavelength,real=real))
        derivs2 = np.array(self.geometries[1].calculate_gradients(gradient_fields, wavelength,real=real))

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

    def get_eps(self,params=None,return_sim=False):
        if params is None:
            params=self.get_current_params()
        sim=self.opt.make_sim()
        eps,x,y,z = ls.get_eps_from_sim(sim.fdtd)
        if return_sim:
            return eps,x,y,z,sim
        else:
            return eps, x, y, z

    def update_geo_in_sim(self, sim, params, eval):
        script = str()
        for geometry in self.geometries:
            script += geometry.update_geo_in_sim(sim, params, eval)
        if eval: sim.fdtd.eval(script)
        return script

    def get_eps_update(self,sim,params=None):
        if params is None:
            params = self.get_current_params()
        sim.fdtd.switchtolayout()
        self.update_geo_in_sim(sim,params, eval = True)
        eps,x,y,z = ls.get_eps_from_sim(sim.fdtd)
        return eps,x,y,z

    def get_d_eps_d_params_update(self,dx):

        current_eps, x, y, z, sim = self.get_eps(return_sim=True)
        current_params=self.get_current_params()
        d_epses = list()
        print('Getting d eps: dx = ' + str(dx))
        for i,param in enumerate(current_params):
            d_params = current_params.copy()
            d_params[i] = param + dx
            d_eps = (self.get_eps_update(sim,d_params)[0]-current_eps) / dx
            d_epses.append(d_eps)
            sys.stdout.write('.'), sys.stdout.flush()
        print('')
        sim.close()
        return d_epses,x,y,z

    def calculate_gradients_from_sims_eps(self, gradient_fields, real = True):
        d_epses, x, y, z = self.get_d_eps_d_params_update(dx = self.dx)
        fields = gradient_fields.sparse_perturbation_field_nosum()
        wl_points = fields.shape[3]
        num_opt_param = len(d_epses)
        derivs = np.zeros((num_opt_param, wl_points), dtype = 'complex')
        for i in range(num_opt_param):
            d_eps = np.take(d_epses[i], indices = 0, axis = 3) # permittivity derivatives are constant over frequency
            for j in range(wl_points):
                derivs[i, j] = trapz3D(np.sum(np.take(fields, indices = j, axis = 3) * d_eps, axis = -1), x, y, z)
        derivs = derivs.squeeze()
        return np.real(derivs) if real else derivs
