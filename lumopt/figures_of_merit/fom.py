import lumopt.lumerical_methods.lumerical_scripts as ls
from numpy import conj,pi

class fom(object):
    '''The Figure of Merit'''

    def __init__(self):
        return

    def get(self):
        return

    def add_adjoint_sources(self):
        return

    def get_wavelengths(self):
        try:
            return [self.wavelength]
        except:
            return self.wavelengths


