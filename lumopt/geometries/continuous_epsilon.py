from geometry import Geometry
from lumopt.utilities.materials import Material
import lumapi
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class ContinousEpsilon2D(Geometry):

    self_update = False

    def __init__(self,eps,x,y,z=0,depth=220e-9,eps_max=3.44**2,eps_min=1.44**2,addmesh=True):
        self.eps=eps
        self.x=x
        self.y=y
        self.z=z
        self.depth=depth
        self.bounds=[(eps_min,eps_max)]*self.eps.size
        self.addmesh=addmesh

    def add_geo(self,sim,params=None):
        handle=sim.solver_handle

        if params is None:
            eps=self.eps
        else:
            eps=np.reshape(params ,(len(self.x),len(self.y)))

        lumapi.putMatrix(handle,'eps_geo',eps)
        lumapi.putMatrix(handle,'x_geo',self.x)
        lumapi.putMatrix(handle,'y_geo',self.y)
        lumapi.putMatrix(handle,'z_geo',[self.z-self.depth/2,self.z+self.depth/2])

        script='addimport;' \
               'temp=zeros(length(x_geo),length(y_geo),2);' \
               'temp(:,:,1)=eps_geo;' \
               'temp(:,:,2)=eps_geo;' \
               'importnk2(sqrt(temp),x_geo,y_geo,z_geo);' \
               'set("detail",1);'

        if self.addmesh:
            mesh_script='addmesh;' \
                        'set("x min",{});' \
                        'set("x max",{});' \
                        'set("y min",{});' \
                        'set("y max",{});' \
                        'set("dx",{});' \
                        'set("dy",{});'.format(np.amin(self.x),np.amax(self.x),np.amin(self.y),np.amax(self.y),self.x[1]-self.x[0],self.y[1]-self.y[0])
            lumapi.evalScript(handle,mesh_script)
        lumapi.evalScript(handle, script)

    def calculate_gradients(self, gradient_fields, wavelength,real=True):

        dx=self.x[1]-self.x[0]
        dy=self.y[1]-self.y[0]
        derivs=[]
        for x in self.x:
            for y in self.y:#,y in zip(xx.reshape(-1),yy.reshape(-1)):
                derivs.append(gradient_fields.integrate_square(center=(x,y),box=(dx,dy),z=self.z,wl=wavelength,real=real))
            print '.',
        print ''
        print 'Done'
        return derivs

    def initialize(self,wavelengths,opt):
        self.opt=opt
        pass

    def update_geometry(self, params): #here params is really just a linearized version of the epsilon map
        self.eps=np.reshape(params ,(len(self.x),len(self.y)))

    def get_current_params(self):
        return np.reshape(self.eps,(-1))

    def plot(self,*args):
        pass


class FunctionDefinedContinuousEpsilon2D(ContinousEpsilon2D):

    def __init__(self,func,initial_params,bounds,z=0,depth=220e-9,dx=1e-15,addmesh=True):
        self.func=func # a function that returns x,y,z,and eps
        self.current_params=initial_params
        self.z=z
        self.depth=depth
        self.eps,self.x,self.y=func(initial_params)
        self.dx=dx
        self.addmesh=addmesh
        self.bounds = bounds

    def update_geometry(self,params):
        self.current_params=params
        self.eps=self.func(self.current_params)[0]

    def get_current_params(self):
        return self.current_params

    def get_linear_eps_for_params(self,params):
        return self.func(params)[0].reshape(-1)


    def get_eps_derivatives(self):
        current_eps=self.eps.reshape(-1)
        eps_derivatives=[]
        for i,param in enumerate(self.current_params):
            d_params=self.current_params.copy()
            d_params[i]=param+self.dx
            d_eps=self.get_linear_eps_for_params(d_params)
            eps_derivatives.append((d_eps-current_eps)/self.dx)
        return eps_derivatives

    def calculate_gradients(self, gradient_fields, wavelength,real=True):
        ''' We have to do a chain rule on this one'''
        deriv_map = super(FunctionDefinedContinuousEpsilon2D,self).calculate_gradients(gradient_fields,wavelength,real=False)
        eps_derivatives=self.get_eps_derivatives()
        derivs=[]
        for eps_derivative in eps_derivatives:
            if real:
                derivs.append(np.real(sum(deriv_map*eps_derivative)))
            else:
                derivs.append(sum(deriv_map*eps_derivative))

        return derivs

    def plot(self,ax=None):
        ax.pcolormesh(self.x,self.y,np.real(self.eps.transpose()))

    def add_geo(self,sim,params=None):
        handle=sim.solver_handle

        if params is None:
            eps=self.eps
        else:
            eps=self.func(self.current_params)[0]

        lumapi.putMatrix(handle,'eps_geo',eps)
        lumapi.putMatrix(handle,'x_geo',self.x)
        lumapi.putMatrix(handle,'y_geo',self.y)
        lumapi.putMatrix(handle,'z_geo',[self.z-self.depth/2,self.z+self.depth/2])

        script='addimport;' \
               'temp=zeros(length(x_geo),length(y_geo),2);' \
               'temp(:,:,1)=eps_geo;' \
               'temp(:,:,2)=eps_geo;' \
               'importnk2(sqrt(temp),x_geo,y_geo,z_geo);' \
               'set("detail",1);'

        if self.addmesh:
            mesh_script='addmesh;' \
                        'set("x min",{});' \
                        'set("x max",{});' \
                        'set("y min",{});' \
                        'set("y max",{});' \
                        'set("dx",{});' \
                        'set("dy",{});'.format(np.amin(self.x),np.amax(self.x),np.amin(self.y),np.amax(self.y),self.x[1]-self.x[0],self.y[1]-self.y[0])
            lumapi.evalScript(handle,mesh_script)
        lumapi.evalScript(handle, script)


class FunctionDefinedContinuousEpsilon3DYeeGrid(Geometry):
    '''Inputs one eps grid, tries to get what the eps will be after Lumerical does the interpolation itself '''


    def __init__(self,func,initial_params,bounds,dx=1e-15,addmesh=True):
        self.func=func # a function that returns eps, x,y,z
        self.current_params=initial_params
        self.eps,self.x,self.y,self.z=func(initial_params)
        self.dx = dx
        self.addmesh = addmesh
        self.bounds = bounds

    def update_geometry(self,params):
        self.current_params=params
        self.eps=self.func(self.current_params)[0]

    def get_current_params(self):
        return self.current_params

    def get_linear_eps_for_params(self,params):
        return self.func(params)[0].reshape(-1)


    def get_eps_derivatives(self):
        '''Returns a list with the derivative of the permittivity on the Yee Grid with respect
        to each parameter'''

        current_eps = self.eps.copy()
        eps_derivatives = []
        for i, param in enumerate(self.current_params):
            d_params = self.current_params.copy()
            d_params[i] = param + self.dx
            d_eps = self.func(d_params)[0]
            eps_derivatives.append((d_eps - current_eps)/self.dx)

        """eps_derivatives are the derivatives for the output of self.func . We need 
        the derivatives on the points of the Yee Cell though since nternally, FDTD interpolate 
        eps onto the Yee Grid. Here I assume that the mesh dimensions are constant in the area
        so things are easier to calculate"""

        eps_derivatives_Yee=[]

        for eps_derivative in eps_derivatives:
            eps_derivative_x=np.zeros(eps_derivative.shape)
            eps_derivative_y=np.zeros(eps_derivative.shape)
            eps_derivative_z=np.zeros(eps_derivative.shape)
            eps_derivative_x[:-1,:,:]=(eps_derivative[:-1,:,:]+eps_derivative[1:,:,:])/2
            eps_derivative_y[:,:-1,:]=(eps_derivative[:,:-1,:]+eps_derivative[:,1:,:])/2
            eps_derivative_z[:,:,:-1]=(eps_derivative[:,:,:-1]+eps_derivative[:,:,1:])/2
            eps_derivatives_Yee.append(np.stack((eps_derivative_x,eps_derivative_y,eps_derivative_z),axis=-1))

        return eps_derivatives_Yee#np.stack((eps_derivatives_x,eps_derivatives_y,eps_derivatives_z),axis=-1)

    def initialize(self,wavelengths,opt):
        self.opt=opt
        pass

    def calculate_gradients(self, gradient_fields, wavelength,real=True):
        ''' We have to do a chain rule on this one'''
        sp_field=gradient_fields.sparse_perturbation_field_nosum[:,:,:,0,:]
        eps_derivatives=self.get_eps_derivatives()
        derivs=[]
        for eps_derivative in eps_derivatives:
            if real:
                derivs.append(np.real(np.sum(sp_field*eps_derivative))*(self.x[1]-self.x[0])*(self.y[1]-self.y[0])*(self.z[1]-self.z[0]))
            else:
                derivs.append(np.sum(sp_field*eps_derivative))*(self.x[1]-self.x[0])*(self.y[1]-self.y[0])*(self.z[1]-self.z[0])
        return derivs

    def plot(self,ax=None):
        pass
        #ax.pcolormesh(self.x,self.y,np.real(self.eps.transpose()))

    def add_geo(self,sim,params=None):
        handle=sim.solver_handle

        if params is None:
            eps=self.eps
        else:
            eps=eps=self.func(self.current_params)[0]

        lumapi.putMatrix(handle,'eps_geo',eps)
        lumapi.putMatrix(handle,'x_geo',self.x)
        lumapi.putMatrix(handle,'y_geo',self.y)
        lumapi.putMatrix(handle,'z_geo',self.z)

        script='addimport;' \
               'importnk2(sqrt(eps_geo),x_geo,y_geo,z_geo);' \
               'set("detail",1);'

        if self.addmesh:
            mesh_script='addmesh;' \
                        'set("x min",{});' \
                        'set("x max",{});' \
                        'set("y min",{});' \
                        'set("y max",{});' \
                        'set("z min",{});' \
                        'set("z max",{});' \
                        'set("dx",{});' \
                        'set("dy",{});' \
                        'set("dz",{});'.format(np.amin(self.x),np.amax(self.x),
                                               np.amin(self.y),np.amax(self.y),
                                               np.amin(self.z),np.amax(self.z),
                                               self.x[1]-self.x[0],self.y[1]-self.y[0],
                                               self.z[1]-self.z[0])

            monitor_script = 'select("opt_fields");' \
                          'set("x min",{});' \
                          'set("x max",{});' \
                          'set("y min",{});' \
                          'set("y max",{});' \
                          'set("z min",{});' \
                          'set("z max",{});'.format(np.amin(self.x), np.amax(self.x),
                                                 np.amin(self.y), np.amax(self.y),
                                                 np.amin(self.z), np.amax(self.z))
            index_script = 'select("opt_fields_index");' \
                          'set("x min",{});' \
                          'set("x max",{});' \
                          'set("y min",{});' \
                          'set("y max",{});' \
                          'set("z min",{});' \
                          'set("z max",{});'.format(np.amin(self.x), np.amax(self.x),
                                                 np.amin(self.y), np.amax(self.y),
                                                 np.amin(self.z), np.amax(self.z))

            lumapi.evalScript(handle,mesh_script+monitor_script+index_script)
        lumapi.evalScript(handle, script)



class FunctionDefinedContinuousEpsilon3DYeeGrid_withoffset(Geometry):
    '''Creates three separate epsilon maps to import into Lumerical'''


    def __init__(self,func,initial_params,bounds,dx=1e-15,addmesh=True):
        self.func=func # a function that returns eps, x,y,z
        self.current_params=initial_params
        self.eps,self.x,self.y,self.z=func(initial_params)
        self.dx = dx
        self.addmesh = addmesh
        self.bounds = bounds

    def update_geometry(self,params):
        self.current_params=params
        self.eps=self.func(self.current_params)[0]

    def get_current_params(self):
        return self.current_params

    # def get_linear_eps_for_params(self,params):
    #     return self.func(params)[0].reshape(-1)

    def get_eps_on_Yee_grid(self,eps):
        '''Interpolates the eps provided onto the Yee Grid (boundaries are a little off)'''

        eps_x = eps.copy()
        eps_y = eps.copy()
        eps_z = eps.copy()
        eps_x[:-1, :, :] = (eps[:-1, :, :] + eps[1:, :, :])/2
        eps_y[:, :-1, :] = (eps[:, :-1, :] + eps[:, 1:, :])/2
        eps_z[:, :, :-1] = (eps[:, :, :-1] + eps[:, :, 1:])/2
        eps_Yee=np.stack((eps_x, eps_y, eps_z), axis=-1)
        return eps_Yee

    def get_eps_derivatives(self):
        '''Returns a list with the derivative of the permittivity on the Yee Grid with respect
        to each parameter'''

        current_eps_Yee = self.get_eps_on_Yee_grid(self.eps)
        eps_derivatives_Yee = []
        for i, param in enumerate(self.current_params):
            d_params = self.current_params.copy()
            d_params[i] = param + self.dx
            d_eps_Yee = self.get_eps_on_Yee_grid(self.func(d_params)[0])
            eps_derivatives_Yee.append((d_eps_Yee - current_eps_Yee)/self.dx)

        return eps_derivatives_Yee#np.stack((eps_derivatives_x,eps_derivatives_y,eps_derivatives_z),axis=-1)

    def initialize(self,wavelengths,opt):
        self.opt=opt
        pass

    def calculate_gradients(self, gradient_fields, wavelength,real=True):
        ''' We have to do a chain rule on this one'''
        sp_field=gradient_fields.sparse_perturbation_field_nosum[:,:,:,0,:]
        eps_derivatives=self.get_eps_derivatives()
        derivs=[]
        for eps_derivative in eps_derivatives:
            if real:
                derivs.append(np.real(np.sum(sp_field*eps_derivative))*(self.x[1]-self.x[0])*(self.y[1]-self.y[0])*(self.z[1]-self.z[0]))
            else:
                derivs.append(np.sum(sp_field*eps_derivative)*(self.x[1]-self.x[0])*(self.y[1]-self.y[0])*(self.z[1]-self.z[0]))
        return derivs

    def plot(self,ax=None):
        pass
        #ax.pcolormesh(self.x,self.y,np.real(self.eps.transpose()))

    def add_geo(self,sim,params=None):
        handle=sim.solver_handle

        if params is None:
            eps=self.eps
        else:
            eps=self.func(self.current_params)[0]
        eps_Yee=self.get_eps_on_Yee_grid(eps)
        lumapi.putMatrix(handle,'eps_geo',eps_Yee)
        lumapi.putMatrix(handle,'x_geo',self.x)
        lumapi.putMatrix(handle,'y_geo',self.y)
        lumapi.putMatrix(handle,'z_geo',self.z)

        script='addimport;' \
               'importnk2(sqrt(eps_geo),x_geo,y_geo,z_geo);' \
               'set("data offset in yee cell",1);' \
               'set("detail",1);'

        if self.addmesh:
            mesh_script='addmesh;' \
                        'set("x min",{});' \
                        'set("x max",{});' \
                        'set("y min",{});' \
                        'set("y max",{});' \
                        'set("z min",{});' \
                        'set("z max",{});' \
                        'set("dx",{});' \
                        'set("dy",{});' \
                        'set("dz",{});'.format(np.amin(self.x),np.amax(self.x),
                                               np.amin(self.y),np.amax(self.y),
                                               np.amin(self.z),np.amax(self.z),
                                               self.x[1]-self.x[0],self.y[1]-self.y[0],
                                               self.z[1]-self.z[0])

            monitor_script = 'select("opt_fields");' \
                          'set("x min",{});' \
                          'set("x max",{});' \
                          'set("y min",{});' \
                          'set("y max",{});' \
                          'set("z min",{});' \
                          'set("z max",{});'.format(np.amin(self.x), np.amax(self.x),
                                                 np.amin(self.y), np.amax(self.y),
                                                 np.amin(self.z), np.amax(self.z))
            index_script = 'select("opt_fields_index");' \
                          'set("x min",{});' \
                          'set("x max",{});' \
                          'set("y min",{});' \
                          'set("y max",{});' \
                          'set("z min",{});' \
                          'set("z max",{});'.format(np.amin(self.x), np.amax(self.x),
                                                 np.amin(self.y), np.amax(self.y),
                                                 np.amin(self.z), np.amax(self.z))

            lumapi.evalScript(handle,mesh_script+monitor_script+index_script)
        lumapi.evalScript(handle, script)


class FunctionDefinedContinuousEpsilon3DYeeGrid_withoffset2(Geometry):
    '''If the function direcly provides data on the Yee grid'''


    def __init__(self,func,initial_params,bounds,dx=1e-15,addmesh=True):
        self.func=func # a function that returns eps, x,y,z
        self.current_params=initial_params
        self.eps,self.x,self.y,self.z=func(initial_params)
        self.dx = dx
        self.addmesh = addmesh
        self.bounds = bounds

    def update_geometry(self,params):
        self.current_params=params
        self.eps=self.func(self.current_params)[0]

    def get_current_params(self):
        return self.current_params

    # def get_linear_eps_for_params(self,params):
    #     return self.func(params)[0].reshape(-1)

    # def get_eps_on_Yee_grid(self,eps):
    #     '''Interpolates the eps provided onto the Yee Grid (boundaries are a little off)'''
    #
    #     eps_x = eps.copy()
    #     eps_y = eps.copy()
    #     eps_z = eps.copy()
    #     eps_x[:-1, :, :] = (eps[:-1, :, :] + eps[1:, :, :])/2
    #     eps_y[:, :-1, :] = (eps[:, :-1, :] + eps[:, 1:, :])/2
    #     eps_z[:, :, :-1] = (eps[:, :, :-1] + eps[:, :, 1:])/2
    #     eps_Yee=np.stack((eps_x, eps_y, eps_z), axis=-1)
    #     return eps_Yee

    def get_eps_derivatives(self):
        '''Returns a list with the derivative of the permittivity on the Yee Grid with respect
        to each parameter'''

        current_eps_Yee = self.eps
        eps_derivatives_Yee = []
        for i, param in enumerate(self.current_params):
            d_params = self.current_params.copy()
            d_params[i] = param + self.dx
            d_eps_Yee = self.func(d_params)[0]
            eps_derivatives_Yee.append((d_eps_Yee - current_eps_Yee)/self.dx)

        return eps_derivatives_Yee#np.stack((eps_derivatives_x,eps_derivatives_y,eps_derivatives_z),axis=-1)

    def initialize(self,wavelengths,opt):
        self.opt=opt
        pass

    def calculate_gradients(self, gradient_fields, wavelength,real=True):
        ''' We have to do a chain rule on this one'''
        sp_field=gradient_fields.sparse_perturbation_field_nosum[:,:,:,0,:]
        eps_derivatives=self.get_eps_derivatives()
        derivs=[]
        for eps_derivative in eps_derivatives:
            if real:
                derivs.append(np.real(np.sum(sp_field*eps_derivative))*(self.x[1]-self.x[0])*(self.y[1]-self.y[0])*(self.z[1]-self.z[0]))
            else:
                derivs.append(np.sum(sp_field*eps_derivative)*(self.x[1]-self.x[0])*(self.y[1]-self.y[0])*(self.z[1]-self.z[0]))
        return derivs

    def plot(self,ax=None):
        pass
        #ax.pcolormesh(self.x,self.y,np.real(self.eps.transpose()))

    def add_geo(self,sim,params=None):
        handle=sim.solver_handle

        if params is None:
            eps=self.eps
        else:
            eps=self.func(self.current_params)[0]
        eps_Yee=self.eps
        lumapi.putMatrix(handle,'eps_geo',eps_Yee)
        lumapi.putMatrix(handle,'x_geo',self.x)
        lumapi.putMatrix(handle,'y_geo',self.y)
        lumapi.putMatrix(handle,'z_geo',self.z)

        script='addimport;' \
               'importnk2(sqrt(eps_geo),x_geo,y_geo,z_geo);' \
               'set("data offset in yee cell",1);' \
               'set("detail",1);'

        if self.addmesh:
            mesh_script='addmesh;' \
                        'set("x min",{});' \
                        'set("x max",{});' \
                        'set("y min",{});' \
                        'set("y max",{});' \
                        'set("z min",{});' \
                        'set("z max",{});' \
                        'set("dx",{});' \
                        'set("dy",{});' \
                        'set("dz",{});'.format(np.amin(self.x),np.amax(self.x),
                                               np.amin(self.y),np.amax(self.y),
                                               np.amin(self.z),np.amax(self.z),
                                               self.x[1]-self.x[0],self.y[1]-self.y[0],
                                               self.z[1]-self.z[0])

            monitor_script = 'select("opt_fields");' \
                          'set("x min",{});' \
                          'set("x max",{});' \
                          'set("y min",{});' \
                          'set("y max",{});' \
                          'set("z min",{});' \
                          'set("z max",{});'.format(np.amin(self.x), np.amax(self.x),
                                                 np.amin(self.y), np.amax(self.y),
                                                 np.amin(self.z), np.amax(self.z))
            index_script = 'select("opt_fields_index");' \
                          'set("x min",{});' \
                          'set("x max",{});' \
                          'set("y min",{});' \
                          'set("y max",{});' \
                          'set("z min",{});' \
                          'set("z max",{});'.format(np.amin(self.x), np.amax(self.x),
                                                 np.amin(self.y), np.amax(self.y),
                                                 np.amin(self.z), np.amax(self.z))

            lumapi.evalScript(handle,mesh_script+monitor_script+index_script)
        lumapi.evalScript(handle, script)






if __name__=='__main__':
    import matplotlib as mpl
    # mpl.use('TkAgg')
    import numpy as np
    # from lumopt.figures_of_merit.modematch_importsource import ModeMatch
    from lumopt.figures_of_merit.modematch import ModeMatch
    from lumopt.optimization import Optimization
    from lumopt.optimizers.generic_optimizers import ScipyOptimizers, FixedStepGradientDescent
    from lumopt.utilities.load_lumerical_scripts import load_from_lsf
    import os
    from lumopt.geometries.polygon import function_defined_Polygon
    from lumopt.utilities.materials import Material
    from lumopt import CONFIG
    import scipy

    script = load_from_lsf(os.path.join(CONFIG['root'], 'examples/staight_waveguide/straight_waveguide.lsf'))

    fom = ModeMatch(modeorder=2, precision=50)
    optimizer = ScipyOptimizers(max_iter=20)
    nx=401
    ny=101
    eps = np.ones((nx, ny))*1.44 ** 2
    eps[90, 10] = 10
    geometry = ContinousEpsilon2D(eps=eps, x=np.linspace(-1e-6, 1e-6, nx), y=np.linspace(-0.4e-6, 0.4e-6, ny))
    # function_defined_Polygon(func=waveguide, initial_params=np.linspace(0.25e-6, 0.25e-6, 10),
    #                                     eps_out=Material(1.44 ** 2), eps_in=Material(2.8 ** 2, 2), bounds=bounds,
    #                                     depth=220e-9,
    #                                     edge_precision=5)

    # geometry=Polygon(eps_in=2.8**2,eps_out=1.44**2)
    opt = Optimization(base_script=script, fom=fom, geometry=geometry, optimizer=optimizer)
    # opt.run()
    ##
    opt.initialize()
    eps,x,y,z=opt.geometry.get_eps()

    x_geo=opt.geometry.x
    y_geo=opt.geometry.y


    print 'ha'