""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import copy
import numpy as np
import scipy as sp
import scipy.optimize as spo

class Optimizer(object):

    def __init__(self,max_iter,scaling_factor=1e6):
        '''
        :param max_iter: Maximum number of iterations to run
        :param scaling_factor: Most optimizers assume the variables to optimize are roughly of order of magnitude ~1.
        Since a lot of the geometry parameters we optimize are of order 1e-9 to 1e-6, it can be a good idea to scale them with
        a multiplicative scaling factor.
        '''
        self.max_iter=max_iter
        self.scaling_factor=scaling_factor

        self.current_fom = None
        self.current_gradients=[]
        self.current_params = []
        self.fom_hist = []
        self.params_hist = []
        self.gradients_hist=[]
        self.iteration = 0

    def initialize(self,start_params,callable_fom,callable_jac,bounds,plotting_function):
        '''Loads the scaled starting point, the bounds and the callables to be used in the optimizer'''

        self.start_point = start_params*self.scaling_factor
        self.callable_fom,self.callable_jac=self.define_callables(callable_fom,callable_jac)
        self.bounds = bounds*self.scaling_factor
        self.define_callback(plotting_function)

    def define_callables(self,callable_fom,callable_jac):
        '''This makes the function that is callable by optimzation methods'''

        def callable_fom_local(params):
            fom=callable_fom(params/self.scaling_factor)
            #self.current_params=params
            #self.current_fom=fom
            return fom


        def callable_jac_local(params):
            gradients= callable_jac(params)/self.scaling_factor
            self.current_gradients=gradients
            return gradients

        return callable_fom_local,callable_jac_local

    def define_callback(self,plotting_function):
        '''A callback that should be called at the end of each iteration'''
        def callback(*args):
            self.params_hist.append(copy.copy(self.current_params))
            self.fom_hist.append(self.current_fom)
            self.gradients_hist.append(copy.copy(self.current_gradients))
            self.iteration += 1
            plotting_function()
            self.report_writing()

        self.callback=callback

    def plot(self,fomax,paramsax,gradients_ax):
        fomax.clear()
        paramsax.clear()
        gradients_ax.clear()
        fomax.plot(range(self.iteration),self.fom_hist)
        fomax.set_xlabel('Iteration')
        fomax.set_title('Figure of Merit')
        fomax.set_ylabel('FOM')
        paramsax.semilogy(range(self.iteration),np.abs(self.params_hist))
        paramsax.set_xlabel('Iteration')
        paramsax.set_ylabel('Parameters')
        paramsax.set_title("Parameter evolution")
        gradients_ax.semilogy(range(self.iteration),np.abs(self.gradients_hist))
        gradients_ax.set_xlabel('Iteration')
        gradients_ax.set_ylabel('Gradient Magnitude')
        gradients_ax.set_title("Gradient evolution")

    def report_writing(self):
        with open('optimization_report.txt','a') as f:

            f.write('AT ITERATION {}:  FOM = {}\n'.format(self.iteration,self.fom_hist[-1]))
            f.write('PARAMETERS = {}\n'.format(self.params_hist[-1]/self.scaling_factor))
            f.write('\n \n')

class ScipyOptimizers(Optimizer):
    '''Using scipy's optimizers to perform the optimizations. Some of the algorithms (L-BFGS-G in particular) can approximate
    the Hessian from the different optimization steps (also called Quasi-Newton Optimization). While this is very powerfull,
    the derivatives calculated here using a continuous adjoint method can be noisy, which in turn can lead to poor behavior of
    these Quasi-Newton methods, which expect machine precision derivatices. Therefore these methods are to be used with some caution.

    Checkout documentation of different optimization methods at  :meth:`scipy.optimize.minimize`

    '''

    def __init__(self,max_iter,method='L-BFGS-B',scaling_factor=1e6,pgtol=1e-9):
        '''
        :param max_iter: maximum number of iterations to run the optimizer for. This is not necessarily equal to the number of times a direct/adjoint simulation pair will be run, since these methods can make several calls for each iteration
        :param method: a string which defines which optimization algorithm to use (see :meth:`scipy.optimize.minimize`)
        :param scaling_factor: See :class:`~lumopt.optimzers.generic_optimizers.Optimizer`. The scaling factor is particularly important for scipy optimizers not to freak out.
        '''
        super(ScipyOptimizers,self).__init__(max_iter,scaling_factor)

        self.method=method
        self.fom_calls=0
        self.pgtol=pgtol

    def define_callables(self,callable_fom,callable_jac):
        '''This makes the function that is callable by scipy's optimization method'''

        def callable_fom_local(params):
            fom=callable_fom(params/self.scaling_factor)
            self.current_params = params
            self.current_fom = fom
            self.fom_calls+=1
            return -fom

        def callable_jac_local(params):
            gradients= -callable_jac(params)/self.scaling_factor
            self.current_gradients = gradients
            if self.fom_calls==1:
                self.callback()
            return gradients

        return callable_fom_local,callable_jac_local

    def run(self):
        print('Running scipy optimizer')
        print('bounds= {}'.format(self.bounds))
        print('start= {}'.format(self.start_point))
        res = spo.minimize(fun=self.callable_fom,x0=self.start_point,jac=self.callable_jac,bounds=self.bounds,callback=self.callback,options={'maxiter':self.max_iter,'disp':True,'gtol':self.pgtol},method=self.method)
        return res
