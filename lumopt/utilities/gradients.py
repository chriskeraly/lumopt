import numpy as np
from scipy.integrate import dblquad,nquad
from lumopt.utilities.scipy_wrappers import dblsimps, wrapped_GridInterpolator
from lumopt.utilities.scipy_wrappers import trapz1D,trapz3D,trapz2D
import matplotlib as mpl
from  lumopt.utilities.fields import Fields

 #mpl.use('TkAgg')
import matplotlib.pyplot as plt
eps0 = 8.854e-12

class Gradient_fields(object):
    '''This class knows how to combine and interpret the forward and adjoint fields in order to
    produce gradient information.

    '''

    def __init__(self,forward_fields,adjoint_fields):
        '''
        :param forward_fields: The forward fields extracted from the forward simulation
        :param adjoint_fields: The adjoint fields extracted from the adjoint simulation
        '''
        self.forward_fields=forward_fields
        self.adjoint_fields=adjoint_fields

        self.sparse_perturbation_field_function_leg=self.def_sparse_function() #legacy
        self.sparse_perturbation_field=None
        self.sparse_perturbation_field_function=self.def_pre_computed_sparse()


    def def_pre_computed_sparse(self):
        '''Interpolations take forever, and a bunch of them need to be done for sparse perturbation. Let's just precompute the damn thing'''
        self.sparse_perturbation_field_nosum=2*eps0*self.forward_fields.E*self.adjoint_fields.E
        self.sparse_perturbation_field=np.sum(2*eps0*self.forward_fields.E*self.adjoint_fields.E,axis=-1)
        interpolator= wrapped_GridInterpolator((self.forward_fields.x, self.forward_fields.y, self.forward_fields.z, self.forward_fields.wl),self.sparse_perturbation_field, method='linear')

        def gradient_field(x,y,z,wl,real=True):
            if real:
                return np.real(interpolator((x,y,z,wl)))[0] #Eq 5.28 of O. Miller's thesis
            else:
                return interpolator((x,y,z,wl))[0] #Eq 5.28 of O. Miller's thesis

        return gradient_field


    def def_sparse_function(self):
        '''Creates a function that simply returns the derivative of the figure of merit against the permittivity at one point in space.
        For the moment this is at a single waveglength

        :returns: gradient_field(x,y,z,wl), a function that returns the derivative of the fom wrt epsilon in space'''

        def gradient_field(x,y,z,wl,real=True):
            #print 'x={} y={}'.format(x,y)
            if real:
                return sum(2*eps0* np.real(self.forward_fields.getfield(x,y,z,wl)*self.adjoint_fields.getfield(x,y,z,wl))) #Eq 5.28 of O. Miller's thesis
            else:
                return sum(2*eps0*self.forward_fields.getfield(x,y,z,wl)*self.adjoint_fields.getfield(x,y,z,wl)) #Eq 5.28 of O. Miller's thesis

        return gradient_field

    def boundary_perturbation_integrand(self,real=True):
        '''Equation 5.28 of Owen's Thesis'''

        def gradient_field(x,y,z,wl,normal,eps_in,eps_out):

            E_forward=self.forward_fields.getfield(x, y, z, wl)
            D_forward=self.forward_fields.getDfield(x, y, z, wl)
            E_adjoint=self.adjoint_fields.getfield(x, y, z, wl)
            D_adjoint=self.adjoint_fields.getDfield(x, y, z, wl)

            def project(a,b):
                '''Project vector a on vector b'''
                b_norm=b/np.linalg.norm(b)
                a1=np.dot(a,b_norm)
                return a1*b_norm

            E_parallel_forward=E_forward-project(E_forward,normal)
            D_perp_forward=project(D_forward,normal)
            E_parallel_adjoint=E_adjoint-project(E_adjoint,normal)
            D_perp_adjoint=project(D_adjoint,normal)

            if real:
                return sum(2* np.real((eps0*eps_in-eps0*eps_out)*E_parallel_forward*E_parallel_adjoint+(1./(eps0*eps_out)-1./(eps0*eps_in))*D_perp_forward*D_perp_adjoint)) #Eq 5.28 of O. Miller's thesis
            else:
                return sum(2* np.array((eps0*eps_in-eps0*eps_out)*E_parallel_forward*E_parallel_adjoint+(1./(eps0*eps_out)-1./(eps0*eps_in))*D_perp_forward*D_perp_adjoint)) #Eq 5.28 of O. Miller's thesis

        return gradient_field

    # def plot(self,original_grid=True,wl=1550e-9):
    #     plt.ion()
    #     if original_grid:
    #         x = self.forward_fields.x
    #         y = self.forward_fields.y
    #     else:
    #         x = np.linspace(min(self.forward_fields.x), max(self.forward_fields.x), 50)
    #         y = np.linspace(min(self.forward_fields.x), max(self.forward_fields.y), 50)
    #     xx, yy = np.meshgrid(x, y)
    #
    #     z = (min(self.forward_fields.z) + max(self.forward_fields.z))/2
    #     Sparse_pert = [self.sparse_perturbation_field_function(x, y, z, wl) for x, y in zip(xx, yy)]
    #     ax = plt.subplot(1, 1, 1)
    #     ax.pcolormesh(xx*1e6, yy*1e6,Sparse_pert,cmap=plt.get_cmap('bwr'))
    #     ax.set_title('Sparse perturbation gradient fields')
    #     plt.show()

    def plot(self,fig,ax_forward,ax_gradients,original_grid=True):
        ax_forward.clear()
        self.forward_fields.plot(ax_forward,title='Forward Fields',cmap='Blues')
        # ax_adjoint.clear()
        # self.adjoint_fields.plot(ax_adjoint,title='Adjoint Fields',cmap='Reds')
        self.plot_gradients(fig,ax_gradients,original_grid)

    def plot_eps(self,ax_eps):
        ax_eps.clear()
        x = self.forward_fields.x
        y = self.forward_fields.y
        eps = self.forward_fields.eps[:,:,0,0,0]
        xx, yy = np.meshgrid(x, y)

        im=ax_eps.pcolormesh(xx*1e6, yy*1e6, np.real(np.transpose(eps)))#, cmap=plt.get_cmap('bwr'))
        ax_eps.set_xlim((np.amin(x)*1e6,np.amax(x)*1e6))
        ax_eps.set_ylim((np.amin(y)*1e6,np.amax(y)*1e6))

        #fig.colorbar(im,ax=ax_gradients)
        ax_eps.set_title('Eps')
        ax_eps.set_xlabel('x(um)')
        ax_eps.set_ylabel('y(um)')

    def plot_gradients(self,fig,ax_gradients,original_grid=True):
        ax_gradients.clear()

        if original_grid:
            x = self.forward_fields.x
            y = self.forward_fields.y
        else:
            x = np.linspace(min(self.forward_fields.x), max(self.forward_fields.x), 50)
            y = np.linspace(min(self.forward_fields.x), max(self.forward_fields.y), 50)
        xx, yy = np.meshgrid(x[1:-1], y[1:-1])

        z = (min(self.forward_fields.z) + max(self.forward_fields.z))/2
        wl = self.forward_fields.wl[0]
        Sparse_pert = [self.sparse_perturbation_field_function_leg(x, y, z, wl) for x, y in zip(xx, yy)]

        im=ax_gradients.pcolormesh(xx*1e6, yy*1e6, Sparse_pert, cmap=plt.get_cmap('bwr'))
        #fig.colorbar(im,ax=ax_gradients)
        ax_gradients.set_title('Sparse perturbation gradient fields')
        ax_gradients.set_xlabel('x(um)')
        ax_gradients.set_ylabel('y(um)')


    def integrate_square(self,center,box,z,wl,real=True):

        xmin=center[0]-box[0]/2.
        xmax=center[0]+box[0]/2.
        ymin=center[1]-box[1]/2.
        ymax=center[1]+box[1]/2.

        integrand=lambda x,y: self.sparse_perturbation_field_function(x,y,z,wl,real=real)

        #res=dblquad(integrand,ymin,ymax,lambda x:xmin,lambda x:xmax,epsrel=1e-3)[0]

        # TODO: speed the integration up, this is making way too many function calls

        #res=nquad(integrand,[(xmin,xmax),(ymin,ymax)],opts=[{'limit':3},{'limit':3}])
        res=dblsimps(integrand,xmin,xmax,ymin,ymax)
        return res

    def integrate_cube(self,center,box,depth,wl,trap_method=True):
        ''' Should this be done with trapz instead?'''

        x_min=center[0]-box[0]/2.
        x_max=center[0]+box[0]/2.
        y_min=center[1]-box[1]/2.
        y_max=center[1]+box[1]/2.
        z_min = center[2] - depth/2.
        z_max = center[2] + depth/2.

        integrand = lambda x, y, z: self.sparse_perturbation_field_function(x, y, z, wl)

        if trap_method:
            z_vect_sim = np.array(self.forward_fields.z)
            z_in_between = z_vect_sim[(z_vect_sim > z_min) & (z_vect_sim < z_max)]
            z_s = np.array([z_min] + list(z_in_between) + [z_max])
            x_vect_sim = np.array(self.forward_fields.x)
            x_in_between = x_vect_sim[(x_vect_sim > x_min) & (x_vect_sim < x_max)]
            x_s = np.array([x_min] + list(x_in_between) + [x_max])
            y_vect_sim = np.array(self.forward_fields.y)
            y_in_between = y_vect_sim[(y_vect_sim > y_min) & (y_vect_sim < y_max)]
            y_s = np.array([y_min] + list(y_in_between) + [y_max])

            vals=[]
            for x in x_s:
                temp_x=[]
                for y in y_s:
                    temp_y=[]
                    for z in z_s:
                        temp_y.append(integrand(x,y,z))
                    temp_x.append(temp_y)
                vals.append(temp_x)
            res =trapz3D(vals,x_s,y_s,z_s)

        else:
            integrand_vals=[]
            for z in np.linspace(z_min,z_max,20):
                integrand=lambda x,y: self.sparse_perturbation_field_function(x,y,z,wl)
                integrand_vals.append(dblsimps(integrand, x_min, x_max, y_min, y_max))
            res=np.trapz(integrand_vals,np.linspace(z_min,z_max,20))

        return res


