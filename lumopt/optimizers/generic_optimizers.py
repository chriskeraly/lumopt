""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import scipy as sp
import scipy.optimize as spo

from lumopt.optimizers.optimizer import Optimizer

class ScipyOptimizers(Optimizer):
    """ Wrapper for the optimizers in SciPy's optimize package: 

            https://docs.scipy.org/doc/scipy/reference/optimize.html#module-scipy.optimize

        Some of the optimization algorithms available in the optimize package ('L-BFGS-G' in particular) can approximate the Hessian from the 
        different optimization steps (also called Quasi-Newton Optimization). While this is very powerfull, the figure of merit gradient calculated 
        from a simulation using a continuous adjoint method can be noisy. This can point Quasi-Newton methods in the wrong direction, so use them 
        with caution.

        Parameters
        ----------
        :param max_iter:       maximum number of iterations; each iteration can make multiple figure of merit and gradient evaluations.
        :param method:         string with the chosen minimization algorithm.
        :param scaling_factor: scalar or a vector of the same length as the optimization parameters; typically used to scale the optimization
                               parameters so that they have magnitudes in the range zero to one.
        :param pgtol:          projected gradient tolerance paramter 'gtol' (see 'BFGS' or 'L-BFGS-G' documentation).
        :param ftol:           tolerance paramter 'ftol' which allows to stop optimization when changes in the FOM are less than this
        :param target_fom:     A target value for the figure of merit. This allows to print/plot the distance of the current
                               design from a target value
        :param scale_initial_gradient_to: 
    """

    def __init__(self, max_iter, method = 'L-BFGS-G', scaling_factor = 1.0, pgtol = 1.0e-5, ftol = 1.0e-12, target_fom = 0, scale_initial_gradient_to = None):
        super(ScipyOptimizers,self).__init__(max_iter, scaling_factor, target_fom, scale_initial_gradient_to)
        self.method = str(method)
        self.fom_calls = int(0)
        self.pgtol = float(pgtol)
        self.ftol=float(ftol)

    def define_callables(self,callable_fom,callable_jac):
        """ Defines the functions that the optimizer will use to evaluate the figure of merit and its gradient. The sign
            of the figure of merit and its gradient are flipped here to perform a maximization rather than a minimization.

            Parameters
            ----------
            :param callable_fom: function taking a numpy vector of optimization parameters and returning the figure of merit.
            :param callable_jac: function taking a numpy vector of optimization parameters and returning a vector of the same size with the figure of merit gradients.
        """

        def callable_fom_local(params):
            fom=callable_fom(params/self.scaling_factor)
            self.current_params = params
            self.current_fom = self.target_fom - fom
            self.fom_calls+=1
            return self.current_fom * self.fom_scaling_factor

        def callable_jac_local(params):
            gradients= -callable_jac(params / self.scaling_factor) / self.scaling_factor
            self.current_gradients = gradients
            if self.fom_calls==1:
                self.callback()
            return gradients * self.fom_scaling_factor

        return callable_fom_local,callable_jac_local

    def run(self):
        print('Running scipy optimizer')
        print('bounds = {}'.format(self.bounds))
        print('start = {}'.format(self.start_point))
        res = spo.minimize(fun = self.callable_fom,
                           x0 = self.start_point,
                           jac = self.callable_jac,
                           bounds = self.bounds,
                           callback = self.callback,
                           options = {'maxiter':self.max_iter, 'disp':True, 'gtol':self.pgtol,'ftol':self.ftol},
                           method = self.method)
        res.x /= self.scaling_factor
        res.fun, res.jac = -res.fun, -res.jac*self.scaling_factor
        return res
