import numpy as np
from lumopt.utilities.scipy_wrappers import trapz1D,trapz3D,trapz2D

class Edge(object):
    '''This class descibes an edge of an extruded 2D geometry'''

    def __init__(self,first_point,second_point,eps_in,eps_out,z=0,depth=220e-9,dimension='2D'):
        self.first_point=first_point
        self.second_point=second_point
        self.eps_in=eps_in
        self.eps_out=eps_out
        self.dimension=dimension
        self.z=z
        self.depth=depth
        if self.dimension=='2D':
            if z is list:
                self.z=z[0]

        self.length=np.sqrt(sum((self.first_point-self.second_point)**2))

        temp = np.flipud((first_point - second_point))
        normal_vect = np.array([-temp[0], temp[1],0])

        self.normal = normal_vect / np.sqrt(sum(normal_vect ** 2))


    def derivative(self,gradient_fields,wl,n_points=5,real=True):

        if len(gradient_fields.forward_fields.z)==1:
            return self.derivative_2D(gradient_fields,wl,n_points=n_points,real=real)
        else:
            return self.derivative_3D(gradient_fields,wl,n_points=n_points,real=real)

    def derivative_3D(self,gradient_fields,wl,n_points,real=True):
        '''Calculates the derivative of an extruded polygon in a 3D simulation by integrating over several layers. The
        layers follow the mesh grid'''

        z_min=self.z-self.depth/2
        z_max=self.z+self.depth/2
        z_vect_sim=np.array(gradient_fields.forward_fields.z)


        z_in_between=z_vect_sim[(z_vect_sim>z_min) & (z_vect_sim<z_max)]
        z_s = np.array([z_min] + list(z_in_between) + [z_max])

        edge_derivative_layers=[self.derivative_2D(gradient_fields,wl,n_points,z_override=z,real=real) for z in z_s]
        deriv_first_integrand=[elem[0] for elem in edge_derivative_layers]
        deriv_second_integrand=[elem[1] for elem in edge_derivative_layers]

        deriv_first=trapz1D(deriv_first_integrand,z_s)
        deriv_second=trapz1D(deriv_second_integrand,z_s)

        return [deriv_first,deriv_second]

    def derivative_2D(self,gradient_fields,wl,n_points,z_override=None,real=True):
        '''Calculates the derivative for moving the two extremity points of an edge in a direction normal to the edge for a
        2D simulation.'''
        #TODO:Maybe this should be done with a dx rather than a number of points? if one could check the mesh size the choice could be automatic rather
        #than rely on the user

        integration_points=[self.first_point*i+self.second_point*(1-i) for i in np.linspace(1,0,n_points)]

        integrand=gradient_fields.boundary_perturbation_integrand(real=real)

        if z_override is None:
            z=self.z
        else:
            z=z_override

        integrand_points=[integrand(point[0],point[1],z,wl,self.normal,self.eps_in.get_eps(wl),self.eps_out.get_eps(wl)) for point in integration_points]

        #calculate derivative for first point:

        weights=np.linspace(1,0,n_points)
        deriv_first=trapz1D(integrand_points*weights,self.length*np.linspace(0,1,n_points))

        # and now the second point
        weights = np.linspace(0, 1, n_points)
        deriv_second=trapz1D(integrand_points*weights,self.length*np.linspace(0,1,n_points))

        derivatives=[deriv_first,deriv_second]
        return derivatives


