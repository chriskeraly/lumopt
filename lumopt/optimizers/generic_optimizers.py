""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import scipy as sp
import scipy.optimize as spo

from lumopt.optimizers.optimizer import Optimizer

class ScipyOptimizers(Optimizer):
    '''Using scipy's optimizers to perform the optimizations. Some of the algorithms (L-BFGS-G in particular) can approximate
    the Hessian from the different optimization steps (also called Quasi-Newton Optimization). While this is very powerfull,
    the derivatives calculated here using a continuous adjoint method can be noisy, which in turn can lead to poor behavior of
    these Quasi-Newton methods, which expect machine precision derivatices. Therefore these methods are to be used with some caution.

    Checkout documentation of different optimization methods at  :meth:`scipy.optimize.minimize`

    '''

    def __init__(self, max_iter, method, scaling_factor, pgtol):
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
