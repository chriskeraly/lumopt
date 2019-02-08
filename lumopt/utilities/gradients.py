""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import numpy as np
import scipy as sp
import scipy.constants
import matplotlib as mpl
import matplotlib.pyplot as plt
from lumopt.utilities.scipy_wrappers import wrapped_GridInterpolator

class Gradient_fields(object):
    """ Combines the forward and adjoint fields (collected by the constructor) to generate the integral kernel 
        used to compute the partial derivatives of the figure of merit with respect to the shape parameters. """

    def __init__(self, forward_fields, adjoint_fields):
        self.forward_fields = forward_fields
        self.adjoint_fields = adjoint_fields

    def sparse_perturbation_field_nosum(self):
        return 2.0 * sp.constants.epsilon_0 * self.forward_fields.E * self.adjoint_fields.E

    def sparse_perturbation_field(self, x, y, z, wl, real = True):
        result = sum(2.0 * sp.constants.epsilon_0 * self.forward_fields.getfield(x,y,z,wl) * self.adjoint_fields.getfield(x,y,z,wl))
        return np.real(result) if real else result

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

    def boundary_perturbation_integrand(self, real):
        ''' Generates the integral kernel in equation 5.28 of Owen Miller's thesis. '''
        def gradient_field(x, y, z, wl, normal, eps_in, eps_out):
            E_forward = self.forward_fields.getfield(x, y, z, wl)
            D_forward = self.forward_fields.getDfield(x, y, z, wl)
            E_adjoint = self.adjoint_fields.getfield(x, y, z, wl)
            D_adjoint = self.adjoint_fields.getDfield(x, y, z, wl)
            def project(a, b):
                b_norm = b / np.linalg.norm(b)
                return np.dot(a,b_norm) * b_norm
            E_parallel_forward = E_forward - project(E_forward, normal)
            D_perp_forward = project(D_forward, normal)
            E_parallel_adjoint = E_adjoint - project(E_adjoint, normal)
            D_perp_adjoint = project(D_adjoint, normal)
            result = sum(2.0 * sp.constants.epsilon_0 * (eps_in - eps_out) * E_parallel_forward * E_parallel_adjoint 
                         + (1.0/eps_out - 1.0/eps_in) / sp.constants.epsilon_0 * D_perp_forward * D_perp_adjoint)
            return np.real(result) if real else result
        return gradient_field