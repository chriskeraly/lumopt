from lumopt import CONFIG
import sys

import lumapi
import lumopt.lumerical_methods.lumerical_scripts as ls


class Simulation(object):
    '''
    Class that handles making and running simulations
    '''

    def __init__(self,workingDir,script=''):
        '''
        :param workingDir: Working directory to save the simulation before running it
        :param script: (String) Base Lumerical script to execute when making the simulation
        '''
        self.script=script
        self.workingDir=workingDir
        self.get_solver_handle()
        lumapi.evalScript(self.solver_handle,script)


    def run(self,name='forward',iter=None):
        '''
        Saves (in the working directory) then runs the simulation with filename 'name_iter.fsp'

        :param name: prefix to the file name
        :param iter: suffix to the file name

        '''
        lumapi.evalScript(self.solver_handle,"save('{}_{}');".format(name,iter))
        lumapi.evalScript(self.solver_handle,'run;')


    def get_gradient_fields(self,monitor_name):
        '''Extracts the fields in the optimizable region. These fields are needed to create the gradient fields'''
        return ls.get_fields(self.solver_handle, monitor_name, get_eps=True, get_D=True,nointerpolation=True)
    
    def clear(self):
        '''Clear everything in a simulation'''
        lumapi.evalScript(self.solver_handle,"switchtolayout;selectall;delete;")
        return

    def switch_to_layout(self):
        lumapi.evalScript(self.solver_handle, "switchtolayout;")

    def remove_sources(self):
        '''Removes all the sources present in a simulation. Is used to create the basis for the adjoint simulation.
         The sources need to have 'source' in their name'''
        lumapi.evalScript(self.solver_handle, "selectpartial('source');delete;")
        return


    def get_solver_handle(self):
        ''' Opens FDTD, goes to the working directory and stores the handle.  '''
        self.solver_handle=lumapi.open('fdtd')
        lumapi.evalScript(self.solver_handle,'cd("{}");'.format(self.workingDir))

    def close(self):
        lumapi.close(self.solver_handle)