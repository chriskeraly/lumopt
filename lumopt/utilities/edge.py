""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import numpy as np
from lumopt.utilities.scipy_wrappers import trapz1D,trapz3D,trapz2D

class Edge(object):
    '''This class descibes an edge of an extruded 2D geometry'''

    def __init__(self, first_point, second_point, eps_in, eps_out, z, depth):
        self.first_point = first_point
        self.second_point = second_point
        self.eps_in = eps_in
        self.eps_out = eps_out
        self.z = float(z)
        self.depth = float(depth)

        normal_vect = np.flipud(first_point - second_point)
        normal_vect = np.array([-normal_vect[0], normal_vect[1],0])
        self.normal = normal_vect / np.sqrt(np.sum(np.power(normal_vect, 2)))

    def derivative(self, gradient_fields, n_points):
        if len(gradient_fields.forward_fields.z) == 1:
            return self.derivative_2D(gradient_fields, n_points = n_points)
        else:
            return self.derivative_3D(gradient_fields, n_points = n_points)

    def derivative_3D(self, gradient_fields, n_points):
        edge_derivs_2D = self.derivative_2D(gradient_fields, n_points)
        deriv_first = edge_derivs_2D[0] * self.depth
        deriv_second = edge_derivs_2D[1] * self.depth
        return (deriv_first, deriv_second)

    def derivative_2D(self, gradient_fields, n_points):
        '''Calculates the derivative for moving the two extremity points of an edge in a direction normal to the edge for a 2D simulation.'''

        # sampling points along edge
        points_along_edge_on_unity_scale = np.linspace(0, 1, n_points)
        points_along_edge_interp_fun = lambda r: self.first_point * (1.0 - r) + self.second_point * r
        points_along_edge = list(map(points_along_edge_interp_fun, points_along_edge_on_unity_scale))
        # integrand in (5.28) of Owen Miller's thesis along the edge
        integrand_interp_func = gradient_fields.boundary_perturbation_integrand()
        wavelengths = gradient_fields.forward_fields.wl
        eps_in = self.eps_in.get_eps(wavelengths)
        eps_out = self.eps_out.get_eps(wavelengths)
        integrand_along_edge = list()
        for idx,wl in enumerate(wavelengths):
            integrand_along_edge_fun = lambda point: integrand_interp_func(point[0], point[1], self.z, wl, self.normal, eps_in[idx], eps_out[idx])
            integrand_along_edge.append(list(map(integrand_along_edge_fun, points_along_edge)))
        integrand_along_edge = np.array(integrand_along_edge).transpose().squeeze()
        # integrate to get derivative at second edge point
        tangent_vec_length = np.sqrt(np.sum(np.power(self.first_point - self.second_point, 2)))
        weights = np.outer(points_along_edge_on_unity_scale, np.ones(len(wavelengths))).squeeze()
        deriv_second = np.trapz(y = integrand_along_edge * weights, x = tangent_vec_length * points_along_edge_on_unity_scale, axis = 0)
        # integrate to get the derivative at first edge point
        flipped_weights = np.flip(weights, axis = 0)
        deriv_first = np.trapz(y = integrand_along_edge * flipped_weights, x = tangent_vec_length * points_along_edge_on_unity_scale, axis = 0)
        # derivatives at both endpoints
        return [deriv_first, deriv_second]
