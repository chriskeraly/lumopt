""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import numpy as np
import scipy as sp
import scipy.constants
import matplotlib as mpl
import matplotlib.pyplot as plt
import lumapi

class GradientFields(object):
    """ Combines the forward and adjoint fields (collected by the constructor) to generate the integral used
        to compute the partial derivatives of the figure of merit (FOM) with respect to the shape parameters. """

    def __init__(self, forward_fields, adjoint_fields):
        assert forward_fields.x.size == adjoint_fields.x.size
        assert forward_fields.y.size == adjoint_fields.y.size
        assert forward_fields.z.size == adjoint_fields.z.size
        assert forward_fields.wl.size == adjoint_fields.wl.size
        self.forward_fields = forward_fields
        self.adjoint_fields = adjoint_fields

    def sparse_perturbation_field(self, x, y, z, wl, real = True):
        result = sum(2.0 * sp.constants.epsilon_0 * self.forward_fields.getfield(x,y,z,wl) * self.adjoint_fields.getfield(x,y,z,wl))
        return np.real(result) if real else result

    def get_field_product_E_forward_adjoint(self):
        
        return self.forward_fields.E*self.adjoint_fields.E


    def get_forward_dot_adjoint_center(self):
        prod = np.sum(2.0 * sp.constants.epsilon_0 * self.get_field_product_E_forward_adjoint(),axis=-1)
        sz = prod.shape
        centerZ = int(sz[2]/2)
        centerLambda = int(sz[3]/2)
        
        return np.transpose(np.real(prod[:,:,centerZ,centerLambda]))
        
    def plot(self, fig, ax_forward, ax_gradients, original_grid = True):
        ax_forward.clear()
        self.forward_fields.plot(ax_forward, title = 'Forward Fields', cmap = 'Blues')
        self.plot_gradients(fig, ax_gradients, original_grid)

    def plot_gradients(self, fig, ax_gradients, original_grid):
        ax_gradients.clear()

        if original_grid:
            x = self.forward_fields.x
            y = self.forward_fields.y
        else:
            x = np.linspace(min(self.forward_fields.x), max(self.forward_fields.x), 50)
            y = np.linspace(min(self.forward_fields.x), max(self.forward_fields.y), 50)
        xx, yy = np.meshgrid(x[1:-1], y[1:-1])

        z = (min(self.forward_fields.z) + max(self.forward_fields.z))/2
        wl = self.forward_fields.wl[0]
        Sparse_pert = [self.sparse_perturbation_field(x, y, z, wl) for x, y in zip(xx, yy)]

        im = ax_gradients.pcolormesh(xx*1e6, yy*1e6, Sparse_pert, cmap = plt.get_cmap('bwr'))
        ax_gradients.set_title('Sparse perturbation gradient fields')
        ax_gradients.set_xlabel('x(um)')
        ax_gradients.set_ylabel('y(um)')

    def plot_eps(self,ax_eps):
        ax_eps.clear()
        x = self.forward_fields.x
        y = self.forward_fields.y
        eps = self.forward_fields.eps[:,:,0,0,0]
        xx, yy = np.meshgrid(x, y)

        im = ax_eps.pcolormesh(xx*1e6, yy*1e6, np.real(np.transpose(eps)))#, cmap=plt.get_cmap('bwr'))
        ax_eps.set_xlim((np.amin(x)*1e6,np.amax(x)*1e6))
        ax_eps.set_ylim((np.amin(y)*1e6,np.amax(y)*1e6))

        #fig.colorbar(im,ax = ax_gradients)
        ax_eps.set_title('Eps')
        ax_eps.set_xlabel('x(um)')
        ax_eps.set_ylabel('y(um)')

    def boundary_perturbation_integrand(self):
        ''' Generates the integral kernel in equation 5.28 of Owen Miller's thesis used to approximate
            the partial derivatives of the FOM with respect to the optimization parameters. '''

        def project(a, b):
            b_norm = b / np.linalg.norm(b)
            return np.dot(a, b_norm) * b_norm

        def gradient_field(x, y, z, wl, normal, eps_in, eps_out):
            ''' All inputs must be scalar numbers. '''
            E_forward = self.forward_fields.getfield(x, y, z, wl)
            D_forward = self.forward_fields.getDfield(x, y, z, wl)
            E_adjoint = self.adjoint_fields.getfield(x, y, z, wl)
            D_adjoint = self.adjoint_fields.getDfield(x, y, z, wl)
            E_parallel_forward = E_forward - project(E_forward, normal)
            E_parallel_adjoint = E_adjoint - project(E_adjoint, normal)
            D_perp_forward = project(D_forward, normal)
            D_perp_adjoint = project(D_adjoint, normal)
            result = ( 2.0 * sp.constants.epsilon_0 * (eps_in - eps_out) * np.sum(E_parallel_forward * E_parallel_adjoint) 
                      + (1.0/eps_out - 1.0/eps_in) / sp.constants.epsilon_0 * np.sum(D_perp_forward * D_perp_adjoint) )
            return np.real(result)
        
        return gradient_field

    @staticmethod
    def spatial_gradient_integral_on_cad(sim, forward_fields, adjoint_fields, wl_scaling_factor):
        lumapi.putMatrix(sim.fdtd.handle, "wl_scaling_factor", wl_scaling_factor)
        sim.fdtd.eval("gradient_fields = 2.0 * eps0 * {0}.E.E * {1}.E.E;".format(forward_fields, adjoint_fields) +
                      "num_opt_params = length(d_epses);" +
                      "num_wl_pts = length({0}.E.lambda);".format(forward_fields) +
                      "partial_fom_derivs_vs_lambda = matrix(num_wl_pts, num_opt_params);" +
                      "for(param_idx = [1:num_opt_params]){"+
                      "    for(wl_idx = [1:num_wl_pts]){" +
                      "        spatial_integrand = pinch(sum(gradient_fields(:,:,:,wl_idx,:) * wl_scaling_factor(wl_idx) * d_epses{param_idx}, 5), 4);" +
                      "        partial_fom_derivs_vs_lambda(wl_idx, param_idx) = integrate2(spatial_integrand, [1,2,3], {0}.E.x, {0}.E.y, {0}.E.z);".format(forward_fields) +
                      "    }" +
                      "}")
        partial_fom_derivs_vs_lambda = lumapi.getVar(sim.fdtd.handle, 'partial_fom_derivs_vs_lambda')
        sim.fdtd.eval("clear(param_idx, wl_idx, num_opt_params, num_wl_pts, spatial_integrand, gradient_fields, wl_scaling_factor, partial_fom_derivs_vs_lambda, d_epses, {0}, {1});".format(forward_fields, adjoint_fields))
        return partial_fom_derivs_vs_lambda