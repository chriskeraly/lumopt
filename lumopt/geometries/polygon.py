""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import sys
import numpy as np
import scipy as sp
import random
import lumapi
from lumopt.geometries.geometry import Geometry
from lumopt.utilities.edge import Edge
from lumopt.utilities.materials import Material


class Polygon(Geometry):
    """ 
        Defines a polygon with vertices on the (x,y)-plane that are extruded along the z direction to create a 3-D shape. The vertices are 
        defined as a numpy array of coordinate pairs np.array([(x0,y0),...,(xn,yn)]). THE VERTICES MUST BE ORDERED IN A COUNTER CLOCKWISE DIRECTION.

        :param points:         array of shape (N,2) defining N polygon vertices.
        :param z:              center of polygon along the z-axis.
        :param depth:          span of polygon along the z-axis.
        :param eps_out:        permittivity of the material around the polygon.
        :param eps_in:         permittivity of the polygon material.
        :param edge_precision: number of quadrature points along each edge for computing the FOM gradient using the shape derivative approximation method.
    """

    def __init__(self, points, z, depth, eps_out, eps_in, edge_precision):
        self.points = points
        self.z = float(z)
        self.depth = float(depth)
        self.edge_precision = int(edge_precision)
        self.eps_out = eps_out if isinstance(eps_out, Material) else Material(eps_out)
        self.eps_in = eps_in if isinstance(eps_in, Material) else Material(eps_in)

        if self.depth <= 0.0:
            raise UserWarning("polygon depth must be positive.")
        if self.edge_precision <= 0:
            raise UserWarning("edge precision must be a positive integer.")

        self.gradients = list()
        self.make_edges()
        self.hash = random.getrandbits(64)

    def make_edges(self):
        '''Creates all the edge objects'''
        edges=[]

        for i,point in enumerate(self.points):
            edges.append(Edge(self.points[i-1],self.points[i],eps_in=self.eps_in,eps_out=self.eps_out,z=self.z,depth=self.depth))
        self.edges=edges

    def use_interpolation(self):
        return False

    def calculate_gradients(self, gradient_fields):
        ''' We calculate gradients with respect to moving each point in x or y direction '''
        self.make_edges()
        print('Calculating gradients for {} edges'.format(len(self.edges)))
        gradient_pairs_edges=[]
        for edge in self.edges:
            gradient_pairs_edges.append(edge.derivative(gradient_fields, n_points = self.edge_precision))
            sys.stdout.write('.')
        print('')
        #the gradients returned for an edge derivative are the gradients with respect to moving each end point perpendicular to that edge
        #This is not exactly what we are looking for here, since we want the derivative w/ respect to moving each point
        #in the x or y direction, so coming up is a lot of projections...

        gradients = list()
        for i,point in enumerate(self.points):
            deriv_edge_1 = gradient_pairs_edges[i][1]
            normal_edge_1 = self.edges[i].normal
            deriv_edge_2 = gradient_pairs_edges[(i+1)%len(self.edges)][0]
            normal_edge_2 = self.edges[(i+1)%len(self.edges)].normal
            deriv_x = np.dot([1,0,0], np.outer(normal_edge_1, deriv_edge_1).squeeze() + np.outer(normal_edge_2, deriv_edge_2).squeeze())
            deriv_y = np.dot([0,1,0], np.outer(normal_edge_1, deriv_edge_1).squeeze() + np.outer(normal_edge_2, deriv_edge_2).squeeze())
            gradients.append(deriv_x)
            gradients.append(deriv_y)
        self.gradients.append(gradients)
        return self.gradients[-1]

    def update_geometry(self, points_linear, sim = None):
        '''Sets the points. Must be fed a linear array of points, because during the optimization the point coordinates are not by pair'''
        self.points =np.reshape(points_linear,(-1,2))

    def get_current_params(self):
        '''returns the points coordinates linearly '''
        return np.reshape(self.points,(-1)).copy()

    def add_geo(self, sim, params, only_update):
        ''' Adds the geometry to a Lumerical simulation'''
        sim.fdtd.switchtolayout()
        if params is None:
            points = self.points
        else:
            points = np.reshape(params, (-1, 2))
        poly_name = 'polygon_{0}'.format(self.hash)
        if not only_update:
            sim.fdtd.addpoly()
            sim.fdtd.set('name', poly_name)
        sim.fdtd.set('x', 0.0)
        sim.fdtd.set('y', 0.0)
        sim.fdtd.set('z', self.z)
        sim.fdtd.set('z span', self.depth)
        sim.fdtd.putv('vertices', points)
        self.eps_in.set_script(sim, poly_name)

    def plot(self,ax):
        points=self.points.copy()
        points=np.reshape(points,(-1,2))
        x_p=points[:,0]*1e6
        y_p=points[:,1]*1e6
        ax.clear()
        ax.plot(x_p,y_p)
        ax.set_title('Geometry')
        ax.set_ylim(min(y_p),max(y_p))
        ax.set_xlim(min(x_p),max(x_p))
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        return True


class FunctionDefinedPolygon(Polygon):
    """ 
        Constructs a polygon from a user defined function that takes the optimization parameters and returns a set of vertices defining a polygon.
        The polygon vertices returned by the function must be defined as a numpy array of coordinate pairs np.array([(x0,y0),...,(xn,yn)]). THE 
        VERTICES MUST BE ORDERED IN A COUNTER CLOCKWISE DIRECTION.

        Parameters
        ----------
        :param fun:            function that takes the optimization parameter values and returns a polygon.
        :param initial_params: initial optimization parameter values.
        :param bounds:         bounding ranges (min/max pairs) for each optimization parameter.
        :param z:              center of polygon along the z-axis.
        :param depth:          span of polygon along the z-axis.
        :param eps_out:        permittivity of the material around the polygon.
        :param eps_in:         permittivity of the polygon material.
        :param edge_precision: number of quadrature points along each edge for computing the FOM gradient using the shape derivative approximation method.
        :param dx:             step size for computing the FOM gradient using permittivity perturbations.
    """

    def __init__(self, func, initial_params, bounds, z, depth, eps_out, eps_in, edge_precision = 5, dx = 1.0e-10):
        self.func = func
        self.current_params = np.array(initial_params).flatten()
        points = func(self.current_params)
        super(FunctionDefinedPolygon, self).__init__(points, z, depth, eps_out, eps_in, edge_precision)
        self.bounds = np.array(bounds)
        self.dx = float(dx)

        if self.bounds.shape[0] != self.current_params.size:
            raise UserWarning("there must be one bound for each parameter.")
        elif self.bounds.shape[1] != 2:
            raise UserWarning("there should be a min and max bound for each parameter.")
        for bound in self.bounds:
            if bound[1] - bound[0] <= 0.0:
                raise UserWarning("bound ranges must be positive.")
        if self.dx <= 0.0:
            raise UserWarning("step size must be positive.")

        self.params_hist = list(self.current_params)

    def update_geometry(self, params, sim = None):
        self.points=self.func(params)
        self.current_params=params
        self.params_hist.append(params)

    def get_current_params(self):
        return self.current_params

    def calculate_gradients(self, gradient_fields):
        polygon_gradients = np.array(Polygon.calculate_gradients(self, gradient_fields))
        polygon_points_linear = self.func(self.current_params).reshape(-1)
        gradients = list()
        for i, param in enumerate(self.current_params):
            d_params = np.array(self.current_params.copy())
            d_params[i] += self.dx
            d_polygon_points_linear = self.func(d_params).reshape(-1)
            partial_derivs = (d_polygon_points_linear - polygon_points_linear) / self.dx
            gradients.append(np.dot(partial_derivs, polygon_gradients))
        self.gradients.append(gradients)
        return np.array(self.gradients[-1])

    def add_poly_script(self, sim, points, only_update):
        poly_name = 'polygon_{}'.format(self.hash)
        if not only_update:
            sim.fdtd.addpoly()
            sim.fdtd.set('name', poly_name)
        sim.fdtd.setnamed(poly_name, 'x', 0.0)
        sim.fdtd.setnamed(poly_name, 'y', 0.0)
        sim.fdtd.setnamed(poly_name, 'z', self.z)
        sim.fdtd.setnamed(poly_name, 'z span', self.depth)
        sim.fdtd.setnamed(poly_name, 'vertices', points)
        self.eps_in.set_script(sim, poly_name)

    def add_geo(self, sim, params, only_update):
        ''' Adds the geometry to a Lumerical simulation'''
        if params is None:
            points = self.points
        else:
            points = self.func(params)
        sim.fdtd.switchtolayout()
        self.add_poly_script(sim, points, only_update)
