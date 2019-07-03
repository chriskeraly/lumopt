""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import numpy as np
import scipy as sp
from lumopt.utilities.scipy_wrappers import wrapped_GridInterpolator
import matplotlib as mpl
import matplotlib.pyplot as plt

class Fields(object):
    """ 
        Container for the raw fields from a field monitor. Several interpolation objects are created internally to evaluate the fields
        at any point in space. Use the auxiliary :method:lumopt.lumerical_methods.lumerical_scripts.get_fields to create this object.
    """

    def __init__(self,x,y,z,wl,E,D,eps,H):

        def process_input(input):
            if type(input) is float:
                input = np.array([input])
            else:
                input = input.squeeze()
            if input.shape == ():
                input = np.array([input])
            return input

        x,y,z,wl=map(process_input,[x,y,z,wl])

        self.x=x
        self.y=y
        self.z=z
        self.E=E
        self.D=D
        self.H=H
        self.wl=wl
        self.eps=eps
        self.pointing_vect=None
        self.normalized=False

        self.getfield=self.make_field_interpolation_object(self.E)
        if not eps is None:
            self.geteps=self.make_field_interpolation_object(self.eps)
        if not D is None:
            self.getDfield=self.make_field_interpolation_object(self.D)
        if not H is None:
            self.getHfield=self.make_field_interpolation_object(self.H)
        self.evals=0

    def scale(self, dimension, factors):
        """
            Scales the E, D and H field arrays along the specified dimension using the provided weighting factors.

            Parameters
            ----------
            :param dimension: 0 (x-axis), 1 (y-axis), 2 (z-axis), (3) frequency and (4) vector component.
            :param factors:   list or vector of weighting factors of the same size as the target field dimension.
        """

        if hasattr(self.E, 'dtype'):
            if self.E.shape[dimension] == len(factors):
                self.E = np.concatenate([np.take(self.E, [index], axis = dimension) * factors[index] for index in range(self.E.shape[dimension])], axis = dimension)
                self.getfield = self.make_field_interpolation_object(self.E)
            else:
                raise UserWarning('number of factors must match the target E-field dimension.')
        if hasattr(self.D, 'dtype'):
            if self.D.shape[dimension] == len(factors):
                self.D = np.concatenate([np.take(self.D, [index], axis = dimension) * factors[index] for index in range(self.D.shape[dimension])], axis = dimension)
                self.getDfield = self.make_field_interpolation_object(self.D)
            else:
                raise UserWarning('number of factors must match the target D-field dimension.')
        if hasattr(self.H, 'dtype'):
            if self.H.shape[dimension] == len(factors):
                self.H = np.concatenate([np.take(self.H, [index], axis = dimension) * factors[index] for index in range(self.H.shape[dimension])], axis = dimension)
                self.getHfield = self.make_field_interpolation_object(self.H)
            else:
                raise UserWarning('number of factors must match the target H-field dimension.')

    def make_field_interpolation_object(self,F):

        wl = self.wl[0] if ((F.shape[3]==1) and (len(self.wl)>1)) else self.wl

        Fx_interpolator = wrapped_GridInterpolator((self.x,self.y,self.z,wl), F[:,:,:,:,0], method='linear', bounds_error = False)
        Fy_interpolator = wrapped_GridInterpolator((self.x,self.y,self.z,wl), F[:,:,:,:,1], method='linear', bounds_error = False)
        Fz_interpolator = wrapped_GridInterpolator((self.x,self.y,self.z,wl), F[:,:,:,:,2], method='linear', bounds_error = False)

        def field_interpolator(x,y,z,wl):
            Fx=Fx_interpolator((x,y,z,wl))
            Fy=Fy_interpolator((x,y,z,wl))
            Fz=Fz_interpolator((x,y,z,wl))

            return np.array((Fx,Fy,Fz)).squeeze() #TODO: fix this! This squeeze is a mistery when matlab is used as midman...

        return field_interpolator

    def plot(self,ax,title,cmap):
        ax.clear()
        xx, yy = np.meshgrid(self.x, self.y)
        z = (min(self.z) + max(self.z))/2 + 1e-10
        wl=self.wl[0]
        E_fields = [self.getfield(x, y, z, wl) for x, y in zip(xx, yy)]
        Ex = np.array([E[0] for E in E_fields])
        Ey = np.array([E[1] for E in E_fields])
        Ez = np.array([E[2] for E in E_fields])

        ax.pcolormesh(xx*1e6, yy*1e6,np.abs(Ex**2+Ey**2+Ez**2) ,cmap=plt.get_cmap(cmap))
        ax.set_title(title+' $E^2$')
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')

    def plot_full(self,D=False,E=True,eps=False,H=False,wl=1550e-9,original_grid=True):

        if E:
            self.plot_field(self.getfield,original_grid=original_grid,wl=wl,name='E')
        if D:
            self.plot_field(self.getDfield, original_grid=original_grid, wl=wl, name='D')
        if eps:
            self.plot_field(self.geteps, original_grid=original_grid, wl=wl, name='eps')
        if H:
            self.plot_field(self.getHfield, original_grid=original_grid, wl=wl, name='H')

    def plot_field(self,field_func=None,original_grid=True,wl=1550e-9,name='field'):
        if field_func is None:
            field_func=self.getfield
        plt.ion()
        if original_grid:
            x = self.x
            y = self.y
        else:
            x = np.linspace(min(self.x), max(self.x), 50)
            y = np.linspace(min(self.y), max(self.y), 50)
        xx, yy = np.meshgrid(x, y)
        z = (min(self.z) + max(self.z))/2+1e-10
        E_fields = [field_func(x, y, z, wl) for x, y in zip(xx, yy)]
        Ex = [E[0] for E in E_fields]
        Ey = [E[1] for E in E_fields]
        Ez = [E[2] for E in E_fields]
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
        if len(self.x) > 1 and len(self.y) > 1:
            ax1.pcolormesh(xx*1e6, yy*1e6, np.real(Ex), cmap=plt.get_cmap('bwr'))
            ax1.set_title('real('+name+'x)')
            ax2.pcolormesh(xx*1e6, yy*1e6, np.real(Ey), cmap=plt.get_cmap('bwr'))
            ax2.set_title('real('+name+'y)')
            ax3.pcolormesh(xx*1e6, yy*1e6, np.real(Ez), cmap=plt.get_cmap('bwr'))
            ax3.set_title('real('+name+'z)')
            f.canvas.draw()
        elif len(self.x) == 1:
            ax1.plot(yy*1e6, np.real(Ex))
            ax1.set_title('real('+name+'x)')
            ax2.plot(yy*1e6, np.real(Ey))
            ax2.set_title('real('+name+'y)')
            ax3.plot(yy*1e6, np.real(Ez))
            ax3.set_title('real('+name+'z)')
            f.canvas.draw()
        else:
            ax1.plot(xx*1e6, np.real(Ex))
            ax1.set_title('real('+name+'x)')
            ax2.plot(xx*1e6, np.real(Ey))
            ax2.set_title('real('+name+'y)')
            ax3.plot(xx*1e6, np.real(Ez))
            ax3.set_title('real('+name+'z)')
            f.canvas.draw()
        plt.show(block=False)

class FieldsNoInterp(Fields):

    def __init__(self,x,y, z, wl, deltas, E ,D, eps, H):

        delta_x = deltas[0]
        delta_y = deltas[1]
        delta_z = deltas[2]

        process_array_shape = lambda input: np.array([input]) if np.isscalar(input) or not any(input.shape) else input.flatten()
        x, y, z, wl, delta_x, delta_y, delta_z = map(process_array_shape, [x, y, z, wl, delta_x, delta_y, delta_z])

        deltas = [delta_x, delta_y, delta_z]

        self.x = x
        self.y = y
        self.z = z
        self.deltas = deltas
        self.E = E
        self.D = D
        self.H = H
        self.wl = wl
        self.eps = eps
        self.pointing_vect = None
        self.normalized = False

        self.getfield = self.make_field_interpolation_object_nointerp(self.E)
        if isinstance(self.eps, np.ndarray):
            self.geteps = self.make_field_interpolation_object_nointerp(self.eps)
        if isinstance(self.D, np.ndarray):
            self.getDfield = self.make_field_interpolation_object_nointerp(self.D)
        if isinstance(self.H, np.ndarray):
            self.getHfield = self.make_field_interpolation_object(self.H)
        self.evals = 0

    def make_field_interpolation_object_nointerp(self,F):

        if( (F.shape[3]==1) and (len(self.wl) > 1) ):
            Fx_interpolator = wrapped_GridInterpolator((self.x + self.deltas[0], self.y, self.z, self.wl[0]), np.take(F, indices = [0], axis = 4), method = 'linear', bounds_error = False)
            Fy_interpolator = wrapped_GridInterpolator((self.x, self.y + self.deltas[1], self.z, self.wl[0]), np.take(F, indices = [1], axis = 4), method = 'linear', bounds_error = False)
            Fz_interpolator = wrapped_GridInterpolator((self.x, self.y, self.z + self.deltas[2], self.wl[0]), np.take(F, indices = [2], axis = 4), method = 'linear', bounds_error = False)
        else:
            Fx_interpolator = wrapped_GridInterpolator((self.x + self.deltas[0], self.y, self.z, self.wl), np.take(F, indices = [0], axis = 4), method = 'linear', bounds_error = False)
            Fy_interpolator = wrapped_GridInterpolator((self.x, self.y + self.deltas[1], self.z, self.wl), np.take(F, indices = [1], axis = 4), method = 'linear', bounds_error = False)
            Fz_interpolator = wrapped_GridInterpolator((self.x, self.y, self.z + self.deltas[2], self.wl), np.take(F, indices = [2], axis = 4), method = 'linear', bounds_error = False)

        def field_interpolator(x, y, z, wl):
            Fx = Fx_interpolator((x, y, z, wl))
            Fy = Fy_interpolator((x, y, z, wl))
            Fz = Fz_interpolator((x, y, z, wl))

            return np.array((Fx, Fy, Fz)).squeeze()

        return field_interpolator

    def plot(self,ax,title,cmap):
        ax.clear()
        xx, yy = np.meshgrid(self.x[1:-1], self.y[1:-1])
        z = (min(self.z) + max(self.z))/2 + 1e-10
        wl=self.wl[0]
        E_fields = [self.getfield(x, y, z, wl) for x, y in zip(xx, yy)]
        Ex = np.array([E[0] for E in E_fields])
        Ey = np.array([E[1] for E in E_fields])
        Ez = np.array([E[2] for E in E_fields])

        ax.pcolormesh(xx*1e6, yy*1e6,np.abs(Ex**2+Ey**2+Ez**2) ,cmap=plt.get_cmap(cmap))
        ax.set_title(title+' $E^2$')
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')

    def scale(self, dimension, factors):
        """
            Scales the E, D and H field arrays along the specified dimension using the provided weighting factors.

            Parameters
            ----------
            :param dimension: 0 (x-axis), 1 (y-axis), 2 (z-axis), (3) frequency and (4) vector component.
            :param factors:   list or vector of weighting factors of the same size as the target field dimension.
        """

        if hasattr(self.E, 'dtype'):
            if self.E.shape[dimension] == len(factors):
                self.E = np.concatenate([np.take(self.E, [index], axis = dimension) * factors[index] for index in range(self.E.shape[dimension])], axis = dimension)
                self.getfield = self.make_field_interpolation_object_nointerp(self.E)
            else:
                raise UserWarning('number of factors must match the target E-field dimension.')
        if hasattr(self.D, 'dtype'):
            if self.D.shape[dimension] == len(factors):
                self.D = np.concatenate([np.take(self.D, [index], axis = dimension) * factors[index] for index in range(self.D.shape[dimension])], axis = dimension)
                self.getDfield = self.make_field_interpolation_object_nointerp(self.D)
            else:
                raise UserWarning('number of factors must match the target D-field dimension.')
        if hasattr(self.H, 'dtype'):
            if self.H.shape[dimension] == len(factors):
                self.H = np.concatenate([np.take(self.H, [index], axis = dimension) * factors[index] for index in range(self.H.shape[dimension])], axis = dimension)
                self.getHfield = self.make_field_interpolation_object(self.H)
            else:
                raise UserWarning('number of factors must match the target H-field dimension.')
