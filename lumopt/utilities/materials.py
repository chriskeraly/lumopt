
from lumopt import CONFIG

import sys


import lumapi

import numpy as np

class Material(object):
    ''' A Material is important because it is what contains the information about the permittivity of the material,
     which is needed to calculate shape derivatives.

     A material can be given in two ways:
      - a float representing the permittivity
      - a string representing the name of a material present in Lumerical's material database (eg 'Si (Silicon) - Palik')

      In the latter case, before an optimization is launched, the material properties at the wavelengths at which the figure
      of merit is defined will have to be extracted from the simulation in the initialization routine.

      :param material: A string (such as "Si (Silicon) - Palik") or a float that represents the material permittivity (3.4**2 for Silicon for example)
      '''
      # TODO: For the moment this doesn't work great with multiple wavelengths


    def __init__(self,material='Si (Silicon) - Palik',mesh_order=None):

        self.material=material
        self.permittivities=None
        self.mesh_order=mesh_order

    def initialize(self, wavelengths=[1550e-9]):
        '''Puts permittivities in self.permittivities using the appropriate method'''

        if type(self.material) is str:
            self.permittivities = self.get_lumerical_permitivities(wavelengths)
        else:# type(self.material) is float or type(self.material) is int:
            self.permittivities =self.material
        # else:
        #     raise ValueError('The material should be defined as a float or a string')

        self.wavelengths=np.array(wavelengths).squeeze()


    def get_lumerical_permitivities(self,wavelengths=[1550e-9]):
        '''Fetches the index from Lumerical if a string was given as the input for the object, and squares it to get
        the permittivity'''

        handle=lumapi.open('fdtd')
        lumapi.putMatrix(handle,'wavelengths',np.array(wavelengths))
        lumapi.evalScript(handle,'permittivities=getindex("{}",c/wavelengths);'.format(self.material))
        indexes= lumapi.getVar(handle,'permittivities')
        lumapi.close(handle)
        permittivities=np.array(indexes)**2
        return permittivities.squeeze()

    def set_script(self):
        '''Script to set the material in Lumerical'''

        if type(self.material) is str:
            script= "set('material','{}');".format(self.material)

        else:
            script="set('index', {});".format(np.sqrt(self.permittivities))

        if not self.mesh_order is None:
            script+="set('override mesh order from material database',1);set('mesh order', {});".format(self.mesh_order)

        return script

    def get_eps(self,wl):
        '''

        :param wl: wavelength at which to get permittivity
        :return: interpolated permittivity
        '''

        if self.wavelengths.size==1:
            return self.permittivities
        else:
            return np.interp(wl,self.wavelengths,self.permittivities)


if __name__=='__main__':
    test=Material()
    test.initialize()
    print test.permittivities
    print test.get_eps(1550e-9)