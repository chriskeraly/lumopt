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
    '''An polygon extruded in the z direction, where the points are allowed to move in any direction in the x-y plane. The points
    and extrusion parameters must be defined, as well as the permittivity (or material) forming the inside of the polygon and the permittivity
    (or material) surrounding the polygon. If the Polygon is surrounded by different materials, the shape derivatives will be wrong along the edges
    where the wrong material surrounds the polygon.


    :param points:
        The points are defined as a numpy array of tupple coordinates np.array([(x0,y0),...,(xn,yn)]). THEY MUST BE DEFINED IN A
        COUNTER CLOCKWISE DIRECTION.
    :param z:
        The center of the polygon along the z axis
    :param depth:
        The depth of the extrusion in the z direction (in meters)
    :param eps_out:
        The permittivity of the outer-material (square of refractive index), or the name of a Lumerical Material, from which the permittivity
        will be extracted. Can also be a Material object from :class:`lumpot.utilities.materials.Material` with a defined mesh order.
    :param eps_in:
        The permittivity of the inner-material (square of refractive index), or the name of a Lumerical Material, from which the permittivity
        will be extracted. Can also be a Material object from :class:`lumpot.utilities.materials.Material` with a defined mesh order.
    :param edge_precision:
        The edges will be discretized when calculating the gradients with respect to moving different points of the geometry. This parmeter
        will define the number of discretization points per edge. It is strongly recommended to have at least a few points per mesh cell.
    '''

    self_update=False

    def __init__(self,points, z, depth, eps_out, eps_in, edge_precision, bounds, dx):
        self.points=points
        self.z=z
        self.depth=depth
        self.gradients=[]
        self.edge_precision=edge_precision
        self.dx=dx
        self.eps_out = eps_out if isinstance(eps_out, Material) else Material(eps_out)
        self.eps_in = eps_in if isinstance(eps_in, Material) else Material(eps_in)
        self.make_edges()
        self.hash = random.getrandbits(64)
        return

    def make_edges(self):
        '''Creates all the edge objects'''
        edges=[]

        for i,point in enumerate(self.points):
            edges.append(Edge(self.points[i-1],self.points[i],eps_in=self.eps_in,eps_out=self.eps_out,z=self.z,depth=self.depth))
        self.edges=edges


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

    def update_geometry(self,points_linear):
        '''Sets the points. Must be fed a linear array of points, because during the optimization the point coordinates are not by pair'''
        self.points =np.reshape(points_linear,(-1,2))

    def get_current_params(self):
        '''returns the points coordinates linearly '''
        return np.reshape(self.points,(-1)).copy()

    def add_geo(self, sim, params = None):
        ''' Adds the geometry to a Lumerical simulation'''

        if params is None:
            points = self.points
        else:
            points = np.reshape(params, (-1, 2))
        sim.fdtd.addpoly()
        poly_name = 'polygon_{0}'.format(self.hash)
        sim.fdtd.set('name', poly_name)
        sim.fdtd.set('x', 0.0)
        sim.fdtd.set('y', 0.0)
        sim.fdtd.set('z', self.z)
        sim.fdtd.set('z span', self.depth)
        sim.fdtd.putv('vertices', points)
        self.eps_in.set_script(sim, poly_name)

    def update_geo_in_sim(self, sim, params):
        points = np.reshape(params, (-1, 2))
        sim.fdtd.select('polygon_{}'.format(self.hash))
        sim.fdtd.set('vertices', points)

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
    '''This defines a polygon from a function that takes the optimization parameters and returns a set of points.

    :param func:
        A function that takes as input a list of optimization parameters and returns a list of point coordinates forming
        the polygon to optimize. See example :func:`~lumpot.geometries.polygon.taper_splitter`.
        The points are defined as a numpy array of tupple coordinates np.array([(x0,y0),...,(xn,yn)]).
        THEY MUST BE DEFINED IN A COUNTER CLOCKWISE DIRECTION.
    :param initial_params:
        The initial parameters, which when fed to the previously defined function, will generate the starting geometry of
        the optimization
    :param Bounds:
        The bounds that should be applied on the optimization parameters
    :param z:
        see :class:`~lumpot.geometries.polygon.Polygon`
    :param depth:
        see :class:`~lumpot.geometries.polygon.Polygon`
    :param eps_out:
        see :class:`~lumpot.geometries.polygon.Polygon`
    :param eps_in:
        see :class:`~lumpot.geometries.polygon.Polygon`
    :param edge_precision:
        see :class:`~lumpot.geometries.polygon.Polygon`
        '''

    def __init__(self, func, initial_params, bounds, z, depth, eps_out, eps_in, edge_precision, dx):
        self.points=func(initial_params)
        self.func=func
        self.z=z
        self.current_params=initial_params
        self.depth=depth
        self.gradients=[]
        self.edge_precision=edge_precision
        self.bounds=bounds
        self.params_hist=[initial_params]
        self.eps_out = eps_out if isinstance(eps_out, Material) else Material(eps_out)
        self.eps_in = eps_in if isinstance(eps_in, Material) else Material(eps_in)
        self.make_edges()
        self.dx=dx
        self.hash = random.getrandbits(128)

    def update_geometry(self,params):
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
        self.add_poly_script(sim, points, only_update)

    def update_geo_in_sim(self, sim, params):
        self.add_geo(sim, params, only_update = True)
