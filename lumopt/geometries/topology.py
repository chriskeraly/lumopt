from lumopt.geometries.geometry import Geometry
from lumopt.utilities.materials import Material
from lumopt.lumerical_methods.lumerical_scripts import set_spatial_interp, get_eps_from_sim

import lumapi
import numpy as np
import scipy as sp
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

eps0 = sp.constants.epsilon_0


class TopologyOptimization2DParameters(Geometry):

    def __init__(self, params, eps_min, eps_max, x, y, z, filter_R, eta, beta):
        self.last_params=params
        self.eps_min=eps_min
        self.eps_max=eps_max
        self.eps = None
        self.x=x
        self.y=y
        self.z=z
        self.bounds=[(0,1)]*(len(x)*len(y))
        self.filter_R = filter_R
        self.eta = eta
        self.beta = beta
        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]
        self.dz = z[1]-z[0] if (hasattr(z, "__len__") and len(z)>1) else 0
        self.depth = z[-1]-z[0] if (hasattr(z, "__len__") and len(z)>1) else 220e-9
        self.beta_factor = 1.2
        self.discreteness = 0

        self.unfold_symmetry = False #< We do not want monitors to unfold symmetry

    def use_interpolation(self):
        return True

    def calc_discreteness(self):
        ''' Computes a measure of discreteness. Is 1 when the structure is completely discrete and less when it is not. '''
        rho = self.calc_params_from_eps(self.eps).flatten()
        return 1 - np.sum(4*rho*(1-rho)) / len(rho)

    def progress_continuation(self):
        self.discreteness = self.calc_discreteness()
        print("Discreteness: {}".format(self.discreteness))

        # If it is sufficiently discrete (99%), we terminate
        if self.discreteness > 0.99:
            return False

        ## Otherwise, we increase beta and keep going
        self.beta *= self.beta_factor
        print('Beta is {}'.format(self.beta))
        return True

    def to_file(self, filename):
        np.savez(filename, params=self.last_params, eps_min=self.eps_min, eps_max=self.eps_max, x=self.x, y=self.y, z=self.z, depth=self.depth, beta=self.beta)

    def calc_params_from_eps(self,eps):
        # Use the permittivity in z-direction. Does not really matter since this is just used for the initial guess and is (usually) heavily smoothed
        return (eps - self.eps_min) / (self.eps_max-self.eps_min)

    def set_params_from_eps(self,eps):
        # Use the permittivity in z-direction. Does not really matter since this is just used for the initial guess and is (usually) heavily smoothed
        self.last_params = self.calc_params_from_eps(eps)

    def extract_parameters_from_simulation(self, sim):
        sim.fdtd.selectpartial('import')
        sim.fdtd.eval('set("enabled",0);')

        sim.fdtd.selectpartial('initial_guess')
        sim.fdtd.eval('set("enabled",1);')
        eps = get_eps_from_sim(sim.fdtd, unfold_symmetry=False)
        sim.fdtd.selectpartial('initial_guess')
        sim.fdtd.eval('set("enabled",0);')

        sim.fdtd.selectpartial('import')
        sim.fdtd.eval('set("enabled",1);')
        reduced_eps = np.real(eps[0])

        self.set_params_from_eps(reduced_eps)


    def get_eps_from_params(self, sim, params):
        rho = np.reshape(params, (len(self.x),len(self.y)))
        self.last_params = rho

        ## Use script function to convert the raw parameters to a permittivity distribution and get the result
        sim.fdtd.putv("topo_rho", rho)
        sim.fdtd.eval(('params = struct;'
                       'params.eps_levels=[{0},{1}];'
                       'params.filter_radius = {2};'
                       'params.beta = {3};'
                       'params.eta = {4};'
                       'params.dx = {5};'
                       'params.dy = {6};'
                       'params.dz = 0.0;'
                       'eps_geo = topoparamstoindex(params,topo_rho);').format(self.eps_min,self.eps_max,self.filter_R,self.beta,self.eta,self.dx,self.dy) )
        eps = sim.fdtd.getv("eps_geo")

        return eps

    def initialize(self, wavelengths, opt):
        self.opt=opt
        pass

    def update_geometry(self, params, sim):
        self.eps = self.get_eps_from_params(sim, params)
        self.discreteness = self.calc_discreteness()

    def get_current_params_inshape(self):
        return self.last_params

    def get_current_params(self):
        params = self.get_current_params_inshape()
        return np.reshape(params,(-1)) if params is not None else None

    def plot(self,ax_eps):
        ax_eps.clear()
        x = self.x
        y = self.y
        eps = self.eps
        ax_eps.imshow(np.real(np.transpose(eps)), vmin=self.eps_min, vmax=self.eps_max, extent=[min(x)*1e6,max(x)*1e6,min(y)*1e6,max(y)*1e6], origin='lower')

        ax_eps.set_title('Eps')
        ax_eps.set_xlabel('x(um)')
        ax_eps.set_ylabel('y(um)')
        return True

    def write_status(self, f):
        f.write(', {:.4f}, {:.4f}'.format(self.beta, self.discreteness))


class TopologyOptimization2D(TopologyOptimization2DParameters):
    '''
    '''
    self_update = False

    def __init__(self, params, eps_min, eps_max, x, y, z=0, filter_R=200e-9, eta=0.5, beta=1):
        super().__init__(params, eps_min, eps_max, x, y, z, filter_R, eta, beta)

    @classmethod
    def from_file(cls, filename, z=0, filter_R=200e-9, eta=0.5, beta = None):
        data = np.load(filename)
        if beta is None:
            beta = data["beta"]
        return cls(data["params"], data["eps_min"], data["eps_max"], data["x"], data["y"], z = z, filter_R = filter_R, eta=eta, beta=beta)

    def set_params_from_eps(self,eps):
        # Use the permittivity in z-direction. Does not really matter since this is just used for the initial guess and is (usually) heavily smoothed
        super().set_params_from_eps(eps[:,:,0,0,2])


    def calculate_gradients_on_cad(self, sim, forward_fields, adjoint_fields, wl_scaling_factor):
        lumapi.putMatrix(sim.fdtd.handle, "wl_scaling_factor", wl_scaling_factor)

        sim.fdtd.eval("V_cell = {};".format(self.dx*self.dy) +
                      "dF_dEps = pinch(sum(2.0 * V_cell * eps0 * {0}.E.E * {1}.E.E,5),3);".format(forward_fields, adjoint_fields) +
                      "num_wl_pts = length({0}.E.lambda);".format(forward_fields) +
                      
                      "for(wl_idx = [1:num_wl_pts]){" +
                      "    dF_dEps(:,:,wl_idx) = dF_dEps(:,:,wl_idx) * wl_scaling_factor(wl_idx);" +
                      "}" + 
                      "dF_dEps = real(dF_dEps);")

        rho = self.get_current_params_inshape()
        sim.fdtd.putv("topo_rho", rho)
        sim.fdtd.eval(('params = struct;'
                       'params.eps_levels=[{0},{1}];'
                       'params.filter_radius = {2};'
                       'params.beta = {3};'
                       'params.eta = {4};'
                       'params.dx = {5};'
                       'params.dy = {6};'
                       'params.dz = 0.0;'
                       'topo_grad = topoparamstogradient(params,topo_rho,dF_dEps);').format(self.eps_min,self.eps_max,self.filter_R,self.beta,self.eta,self.dx,self.dy) )
        topo_grad = sim.fdtd.getv("topo_grad")

        return topo_grad.reshape(-1, topo_grad.shape[-1])


    def calculate_gradients(self, gradient_fields, sim):

        rho = self.get_current_params_inshape()

        # If we have frequency data (3rd dim), we need to adjust the dimensions of epsilon for broadcasting to work
        E_forward_dot_E_adjoint = np.atleast_3d(np.real(np.squeeze(np.sum(gradient_fields.get_field_product_E_forward_adjoint(),axis=-1))))

        dF_dEps = 2*self.dx*self.dy*eps0*E_forward_dot_E_adjoint
        
        sim.fdtd.putv("topo_rho", rho)
        sim.fdtd.putv("dF_dEps", dF_dEps)
        sim.fdtd.eval(('params = struct;'
                       'params.eps_levels=[{0},{1}];'
                       'params.filter_radius = {2};'
                       'params.beta = {3};'
                       'params.eta = {4};'
                       'params.dx = {5};'
                       'params.dy = {6};'
                       'params.dz = 0.0;'
                       'topo_grad = topoparamstogradient(params,topo_rho,dF_dEps);').format(self.eps_min,self.eps_max,self.filter_R,self.beta,self.eta,self.dx,self.dy) )
        topo_grad = sim.fdtd.getv("topo_grad")

        return topo_grad.reshape(-1, topo_grad.shape[-1])


    def add_geo(self, sim, params=None, only_update = False):

        fdtd=sim.fdtd

        eps = self.eps if params is None else self.get_eps_from_params(sim, params.reshape(-1))

        fdtd.putv('x_geo',self.x)
        fdtd.putv('y_geo',self.y)
        fdtd.putv('z_geo',np.array([self.z-self.depth/2,self.z+self.depth/2]))

        if not only_update:
            set_spatial_interp(sim.fdtd,'opt_fields','specified position') 
            set_spatial_interp(sim.fdtd,'opt_fields_index','specified position') 

            script=('select("opt_fields");'
                    'set("x min",{});'
                    'set("x max",{});'
                    'set("y min",{});'
                    'set("y max",{});').format(np.amin(self.x),np.amax(self.x),np.amin(self.y),np.amax(self.y))
            fdtd.eval(script)

            script=('select("opt_fields_index");'
                    'set("x min",{});'
                    'set("x max",{});'
                    'set("y min",{});'
                    'set("y max",{});').format(np.amin(self.x),np.amax(self.x),np.amin(self.y),np.amax(self.y))
            fdtd.eval(script)

            script=('addimport;'
                    'set("detail",1);')
            fdtd.eval(script)

            mesh_script=('addmesh;'
                        'set("x min",{});'
                        'set("x max",{});'
                        'set("y min",{});'
                        'set("y max",{});'
                        'set("dx",{});'
                        'set("dy",{});').format(np.amin(self.x),np.amax(self.x),np.amin(self.y),np.amax(self.y),self.dx,self.dy)
            fdtd.eval(mesh_script)

        if eps is not None:
            fdtd.putv('eps_geo',eps)

            ## We delete and re-add the import to avoid a warning
            script=('select("import");'
                    'delete;'
                    'addimport;'
                    'temp=zeros(length(x_geo),length(y_geo),2);'
                    'temp(:,:,1)=eps_geo;'
                    'temp(:,:,2)=eps_geo;'
                    'importnk2(sqrt(temp),x_geo,y_geo,z_geo);')
            fdtd.eval(script)



## Uses a continuous parameter rho in [0,1] instead of the actual epsilon values as parameters. Makes it
## easier to introduce penalization.
class TopologyOptimization3DLayered(TopologyOptimization2DParameters):

    self_update = False

    def __init__(self, params, eps_min, eps_max, x, y, z, filter_R = 200e-9, eta = 0.5, beta = 1):
        super(TopologyOptimization3DLayered,self).__init__(params, eps_min, eps_max, x, y, z, filter_R, eta, beta)

    @classmethod
    def from_file(cls, filename, p, filter_R, eta, beta = None):
        data = np.load(filename)
        if beta is None:
            beta = data["beta"]
        return cls(data["params"], data["eps_min"], data["eps_max"], data["x"], data["y"], data["z"], filter_R = filter_R, eta=eta, beta=beta)

    def to_file(self, filename):
        np.savez(filename, params=self.last_params, eps_min=self.eps_min, eps_max=self.eps_max, x=self.x, y=self.y, z=self.z, beta=self.beta)

    def set_params_from_eps(self,eps):
        '''
            The raw epsilon of a 3d system needs to be collapsed to 2d first. For now, we just pick the first z-layer
        '''
        midZ_idx=int((eps.shape[2]+1)/2)
        super().set_params_from_eps(eps[:,:,midZ_idx,0,2])

    def calculate_gradients(self, gradient_fields, sim):

        rho = self.get_current_params_inshape()

        ## Perform the dot-product which corresponds to a sum over the last dimension (which is x,y,z-components)
        E_forward_dot_E_adjoint = np.real(np.squeeze(np.sum(gradient_fields.get_field_product_E_forward_adjoint(),axis=-1)))
 
        ## We integrate/sum along the z-direction
        E_forward_dot_E_adjoint_int_z = np.atleast_3d(np.squeeze(np.sum(E_forward_dot_E_adjoint,axis=2)))

        V_cell = self.dx*self.dy*self.dz
        dF_dEps = 2*V_cell*eps0*E_forward_dot_E_adjoint_int_z

        sim.fdtd.putv("topo_rho", rho)
        sim.fdtd.putv("dF_dEps", dF_dEps)
        sim.fdtd.eval(('params = struct;'
                       'params.eps_levels=[{0},{1}];'
                       'params.filter_radius = {2};'
                       'params.beta = {3};'
                       'params.eta = {4};'
                       'params.dx = {5};'
                       'params.dy = {6};'
                       'params.dz = 0.0;'
                       'topo_grad = topoparamstogradient(params,topo_rho,dF_dEps);').format(self.eps_min,self.eps_max,self.filter_R,self.beta,self.eta,self.dx,self.dy) )
        topo_grad = sim.fdtd.getv("topo_grad")

        return topo_grad.reshape(-1, topo_grad.shape[-1])


    def calculate_gradients_on_cad(self, sim, forward_fields, adjoint_fields, wl_scaling_factor):
        lumapi.putMatrix(sim.fdtd.handle, "wl_scaling_factor", wl_scaling_factor)

        sim.fdtd.eval("V_cell = {};".format(self.dx*self.dy*self.dz) +
                      "dF_dEps = sum(sum(2.0 * V_cell * eps0 * {0}.E.E * {1}.E.E,5),3);".format(forward_fields, adjoint_fields) +
                      "num_wl_pts = length({0}.E.lambda);".format(forward_fields) +
                      
                      "for(wl_idx = [1:num_wl_pts]){" +
                      "    dF_dEps(:,:,wl_idx) = dF_dEps(:,:,wl_idx) * wl_scaling_factor(wl_idx);" +
                      "}" + 
                      "dF_dEps = real(dF_dEps);")

        rho = self.get_current_params_inshape()
        sim.fdtd.putv("topo_rho", rho)
        sim.fdtd.eval(('params = struct;'
                       'params.eps_levels=[{0},{1}];'
                       'params.filter_radius = {2};'
                       'params.beta = {3};'
                       'params.eta = {4};'
                       'params.dx = {5};'
                       'params.dy = {6};'
                       'params.dz = 0.0;'
                       'topo_grad = topoparamstogradient(params,topo_rho,dF_dEps);').format(self.eps_min,self.eps_max,self.filter_R,self.beta,self.eta,self.dx,self.dy) )
        topo_grad = sim.fdtd.getv("topo_grad")

        return topo_grad.reshape(-1, topo_grad.shape[-1])

    def add_geo(self, sim, params=None, only_update = False):

        fdtd=sim.fdtd

        eps = self.eps if params is None else self.get_eps_from_params(sim, params.reshape(-1))

        if not only_update:
            set_spatial_interp(sim.fdtd,'opt_fields','specified position') 
            set_spatial_interp(sim.fdtd,'opt_fields_index','specified position') 

            script=('select("opt_fields");'
                    'set("x min",{});'
                    'set("x max",{});'
                    'set("y min",{});'
                    'set("y max",{});'
                    'set("z min",{});'
                    'set("z max",{});').format(np.amin(self.x),np.amax(self.x),np.amin(self.y),np.amax(self.y),np.amin(self.z),np.amax(self.z))
            fdtd.eval(script)

            script=('select("opt_fields_index");'
                    'set("x min",{});'
                    'set("x max",{});'
                    'set("y min",{});'
                    'set("y max",{});'
                    'set("z min",{});'
                    'set("z max",{});').format(np.amin(self.x),np.amax(self.x),np.amin(self.y),np.amax(self.y),np.amin(self.z),np.amax(self.z))
            fdtd.eval(script)

            script=('addimport;'
                    'set("detail",1);')
            fdtd.eval(script)

            mesh_script=('addmesh;'
                        'set("x min",{});'
                        'set("x max",{});'
                        'set("y min",{});'
                        'set("y max",{});'
                        'set("z min",{});'
                        'set("z max",{});'
                        'set("dx",{});'
                        'set("dy",{});'
                        'set("dz",{});').format(np.amin(self.x),np.amax(self.x),np.amin(self.y),np.amax(self.y),np.amin(self.z),np.amax(self.z),self.dx,self.dy,self.dz)
            fdtd.eval(mesh_script)


        if eps is not None:
            # This is a layer geometry, so we need to expand it to all layers
            full_eps = np.broadcast_to(eps[:, :, None],(len(self.x),len(self.y),len(self.z)))   #< TODO: Move to Lumerical script to reduce transfers

            fdtd.putv('x_geo',self.x)
            fdtd.putv('y_geo',self.y)
            fdtd.putv('z_geo',self.z)
            fdtd.putv('eps_geo',full_eps)

            ## We delete and re-add the import to avoid a warning
            script=('select("import");'
                    'delete;'
                    'addimport;'
                    'importnk2(sqrt(eps_geo),x_geo,y_geo,z_geo);')
            fdtd.eval(script)
