import numpy as np
import scipy.optimize as sci
import copy

class Optimizer(object):

    done=False

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
        self.iteration=0

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
            self.iteration+=1
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


class FixedStepGradientDescent(Optimizer):
    r'''Simple gradient descent with the option to add noise, and parameter scaling

    Update Equation:

    .. math::
        \Delta p_i = \frac{\frac{dFOM}{dp_i}}{max_j(|\frac{dFOM}{dp_j}|)}\Delta x +\text{noise}_i

    if all_params_equal = True

    .. math::
        \Delta p_i = sign(\frac{dFOM}{dp_i})\Delta x +\text{noise}_i

    Noise can be added in the update equation, if the optimization has many local optima:
    (If no noise is desired set noise_magnitude to 0)
    noise = rand([-1,1])*noise_magnitude
    '''
    done = False

    def __init__(self, max_dx, max_iter, all_params_equal=False, noise_magnitude=0,scaling_factor=1):
        '''

        :param max_dx: Maximum allowed change of a parameter per iteration
        :param max_iter: Maximum number of iterations to run
        :param all_params_equal: If true, all parameters will be changed by +/- dx depending on the sign of their associated
        shape derivative.
        :param noise_magnitude: The amplitude of the noise
        :param scaling_factor: Scaling factor to bring the optimization variables closer to 1

        '''

        super(FixedStepGradientDescent,self).__init__(max_iter,scaling_factor)

        self.max_dx = max_dx*self.scaling_factor
        self.all_params_equal = all_params_equal
        self.noise_magnitude = noise_magnitude*self.scaling_factor
        self.predictedchange_hist = []



    def run(self,verbose=True):
        self.current_params=self.start_point
        while self.iteration<self.max_iter:
            if verbose:
                print('Startin iteration number {}'.format(self.iteration))
            current_fom=self.callable_fom(self.current_params)
            self.current_fom=current_fom
            if verbose:
                print('New figure of merit: {}'.format(current_fom))
            gradients=self.callable_jac(self.current_params)

            change=self.calculate_change(gradients,self.max_dx)
            self.current_params += change


            self.add_noise()
            self.current_params=self.enforce_bounds(self.current_params)

            self.predictedchange_hist.append(sum(gradients*change))

            self.callback()

    def calculate_change(self,gradients,dx):
        if self.all_params_equal:
            change = ((np.array(gradients) > 0)*2 - 1)*dx
        else:
            change = (np.array(gradients)/max(abs(np.array(gradients))))*dx
        return change

    def add_noise(self):
        noise = self.noise_magnitude*(np.random.rand(len(self.current_params)) - 0.5)*2
        self.current_params = self.current_params + noise

    def enforce_bounds(self,params):
        bounds_min = np.array([bound[0] for bound in self.bounds])
        bounds_max = np.array([bound[1] for bound in self.bounds])
        return np.maximum(bounds_min, (np.minimum(bounds_max, params)))


class Adaptive_Gradient_Descent(FixedStepGradientDescent):
    ''' Same as :class:`~lumopt.optimizers.generic_optimizers.FixedStepGradientDescent`, except that dx changes according to these rules for each iteration:

    dx=min(max_dx,dx*dx_regrowth_factor)\n
    while newfom<oldfom\n
        dx=dx/2\n
        if dx<min_dx:\n
            dx=min_dx\n
            return newfom # forces an update even though the figure of merit went down'''

    done = False

    def __init__(self, max_dx, min_dx, max_iter, dx_regrowth_factor=1.1, all_params_equal=False, noise_magnitude=0,scaling_factor=1):
        '''

        :param max_dx: Maximum allowed change of a parameter per iteration
        :param min_dx: Minimum step size (for the largest parameter changing) allowed
        :param dx_regrowth_factor: by how much dx will be increased at each iteration
        :param max_iter: Maximum number of iterations to run
        :param all_params_equal: If true, all parameters will be changed by +/- dx depending on the sign of their associated
        shape derivative.
        :param noise_magnitude: The amplitude of the noise
        :param scaling_factor: Scaling factor to bring the optimization variables closer to 1
        '''

        super(FixedStepGradientDescent,self).__init__(max_iter,scaling_factor)

        self.max_dx = max_dx*self.scaling_factor
        self.all_params_equal = all_params_equal
        self.noise_magnitude = noise_magnitude*self.scaling_factor
        self.predictedchange_hist = []
        self.min_dx=min_dx*self.scaling_factor
        self.dx_regrowth_factor=dx_regrowth_factor
        self.dx=self.max_dx

    def run(self, verbose = True):
        self.current_params = self.start_point

        #First iteration outside the loop
        print('Startin iteration number {}'.format(self.iteration))
        self.current_fom = self.callable_fom(self.current_params)
        if verbose:
            print('New figure of merit: {}'.format(self.current_fom))
        gradients = self.callable_jac(self.current_params)

        self.callback() #sets fomhist and such


        while self.iteration < self.max_iter:

            self.dx=np.minimum(self.dx*self.dx_regrowth_factor,self.max_dx)

            if verbose:
                print('Startin iteration number {}'.format(self.iteration))

            new_params=self.current_params+self.calculate_change(gradients,self.dx)
            new_params=self.enforce_bounds(new_params)
            new_fom = self.callable_fom(new_params)

            while new_fom<self.fom_hist[-1] and self.dx>self.min_dx:
                self.reduce_step_size()
                new_params = self.current_params + self.calculate_change(gradients, self.dx)
                new_params=self.enforce_bounds(new_params)
                new_fom = self.callable_fom(new_params)

            if self.dx==self.min_dx:
                print('dx at mindx: forcing update')

            self.current_params=new_params
            self.current_fom=new_fom

            if verbose:
                print('New figure of merit: {}'.format(self.current_fom))

            if not self.iteration==self.max_iter:
                gradients = self.callable_jac(self.current_params)
            self.callback()




    def reduce_step_size(self):
        self.dx=np.maximum(self.dx/2,self.min_dx)
        print('Figure of merit decreasing: reducing step size to {}'.format(self.dx))




class ScipyOptimizers(Optimizer):
    '''Using scipy's optimizers to perform the optimizations. Some of the algorithms (L-BFGS-G in particular) can approximate
    the Hessian from the different optimization steps (also called Quasi-Newton Optimization). While this is very powerfull,
    the derivatives calculated here using a continuous adjoint method can be noisy, which in turn can lead to poor behavior of
    these Quasi-Newton methods, which expect machine precision derivatices. Therefore these methods are to be used with some caution.

    Checkout documentation of different optimization methods at  :meth:`scipy.optimize.minimize`

    '''
    #TODO: implement a good callback to plot and save all the stuff
    done=False


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


    def run(self,verbose=True):
        print('Running Scipy Optimizer')
        print('bounds= {}'.format(self.bounds))
        print('start= {}'.format(self.start_point))
        res= sci.minimize(fun=self.callable_fom,x0=self.start_point,jac=self.callable_jac,bounds=self.bounds,callback=self.callback,options={'maxiter':self.max_iter,'disp':True,'gtol':self.pgtol},method=self.method)
        return res
