""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import numpy as np
import scipy as sp
import scipy.constants

from lumopt.utilities.wavelengths import Wavelengths

class Material(object):
    ''' Permittivity of a material associated with a geometric primitive.

        In FDTD Solutions, a material can be given in two ways:

            1) By providing a material name from the material database (e.g. 'Si (Silicon) - Palik') that can be assigned to a geometric primitive.
            2) By providing a refractive index value directly in geometric primitive.

        To use the first option, simply set the name to '<Object defined dielectric>' and enter the desired base permittivity value.
        To use the second option, set the name to the desired material name and the base permittivity to none.

        :name:         string (such as "Si (Silicon) - Palik") with a valid material name.
        :base_epsilon: scalar base permittivity value.
        :mesh_order:   order of material resolution for overlapping primitives.
      '''

    object_dielectric = str('<Object defined dielectric>')

    def __init__(self, base_epsilon = 1.0, name = object_dielectric, mesh_order = None):
        self.base_epsilon = float(base_epsilon)
        self.name = str(name)
        self.mesh_order = mesh_order

    def set_script(self, sim, poly_name):
        sim.fdtd.setnamed(poly_name, 'material', self.name)
        self.wavelengths = Material.get_wavelengths(sim)
        freq_array = sp.constants.speed_of_light / self.wavelengths.asarray()
        if self.name == self.object_dielectric:
            refractive_index = np.sqrt(self.base_epsilon)
            sim.fdtd.setnamed(poly_name, 'index', float(refractive_index))
            self.permittivity = self.base_epsilon * np.ones(freq_array.shape)
        else:
            fdtd_index = sim.fdtd.getfdtdindex(self.name, freq_array, float(freq_array.min()), float(freq_array.max()))
            self.permittivity = np.asarray(np.power(fdtd_index, 2)).flatten()
        if self.mesh_order:
            sim.fdtd.setnamed(poly_name, 'override mesh order from material database', True)
            sim.fdtd.setnamed(poly_name, 'mesh order', self.mesh_order)

    def get_eps(self, wavelengths):
        if hasattr(self, 'permittivity'):
            assert len(wavelengths) == len(self.wavelengths) # should be identical
            return self.permittivity
        elif self.name == self.object_dielectric:
            return self.base_epsilon * np.ones(wavelengths.shape)
        else:
            raise UserWarning('material has not yet been assigned to a geometric primitive.')

    @staticmethod
    def get_wavelengths(sim):
        return Wavelengths(sim.fdtd.getglobalsource('wavelength start'), 
                           sim.fdtd.getglobalsource('wavelength stop'),
                           sim.fdtd.getglobalmonitor('frequency points'))
