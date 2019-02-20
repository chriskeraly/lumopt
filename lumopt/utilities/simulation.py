""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

from lumopt import CONFIG
import sys
import lumapi
import lumopt.lumerical_methods.lumerical_scripts as ls


class Simulation(object):
    '''
    Class that handles making and running simulations
    '''

    def __init__(self, workingDir, script, hide_fdtd_cad):
        '''
        :param workingDir: Working directory to save the simulation before running it
        :param script: (String) Base Lumerical script to execute when making the simulation
        '''
        self.fdtd = lumapi.FDTD(hide = hide_fdtd_cad)
        self.fdtd.cd(workingDir)
        self.fdtd.eval(script)

    def run(self,name='forward',iter=None):
        '''
        Saves (in the working directory) then runs the simulation with filename 'name_iter.fsp'

        :param name: prefix to the file name
        :param iter: suffix to the file name

        '''
        self.fdtd.save('{}_{}'.format(name,iter))
        self.fdtd.run()


    def get_gradient_fields(self,monitor_name):
        '''Extracts the fields in the optimizable region. These fields are needed to create the gradient fields'''
        return ls.get_fields(self.fdtd, monitor_name, get_eps = True, get_D = True, get_H = False, nointerpolation = True)

    def remove_sources(self):
        '''Removes all the sources present in a simulation. Is used to create the basis for the adjoint simulation.
         The sources need to have 'source' in their name'''
        self.fdtd.selectpartial('source')
        self.fdtd.delete()
        return


    def close(self):
        self.fdtd.close()
