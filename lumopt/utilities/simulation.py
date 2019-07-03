""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import lumapi

class Simulation(object):
    """
        Object to manage the FDTD CAD. 

        Parameters
        ----------
        :param workingDir:    working directory for the CAD session.
        :param hide_fdtd_cad: if true, runs the FDTD CAD in the background.
    """

    def __init__(self, workingDir, use_var_fdtd, hide_fdtd_cad):
        """ Launches FDTD CAD and stores a handle. """
        self.fdtd = lumapi.MODE(hide = hide_fdtd_cad) if use_var_fdtd else lumapi.FDTD(hide = hide_fdtd_cad)
        self.workingDir = workingDir
        self.fdtd.cd(self.workingDir)

    def run(self, name, iter):
        """ Saves simulation file and runs the simulation. """
        self.fdtd.cd(self.workingDir)
        self.fdtd.save('{}_{}'.format(name,iter))
        self.fdtd.run()

    def remove_data_and_save(self):
        self.fdtd.switchtolayout()
        self.fdtd.save()
        
    def __del__(self):
        self.fdtd.close()
