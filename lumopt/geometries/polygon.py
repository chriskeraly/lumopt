from lumopt.geometries.geometry import Geometry
import numpy as np
from lumopt.utilities.edge import Edge
import scipy
from lumopt.utilities.materials import Material


import lumapi
import random
#import matlab

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

    def __init__(self,points=np.array([(1,1),(-1,1),(-1,-1),(1,-1)])*0.05e-6,z=0,depth=100e-9,eps_out=2**2,eps_in=3.44**2,edge_precision=10,bounds=None,dx=1e-9):
        #given properties
        self.points=points
        self.z=z
        self.depth=depth
        #self.index=np.sqrt(eps_in)
        self.gradients=[]
        self.edge_precision=edge_precision
        self.dx=dx
        if type(eps_out) is Material:
            self.eps_out=eps_out
        else:
            self.eps_out=Material(eps_out)
        if type(eps_in) is Material:
            self.eps_in = eps_in
        else:
            self.eps_in = Material(eps_in)
        self.make_edges()
        self.hash=random.getrandbits(128)
        return

    def make_edges(self):
        '''Creates all the edge objects'''
        edges=[]

        for i,point in enumerate(self.points):
            edges.append(Edge(self.points[i-1],self.points[i],eps_in=self.eps_in,eps_out=self.eps_out,z=self.z,depth=self.depth))
        self.edges=edges


    def calculate_gradients(self,gradient_fields,wavelength,real=True):
        ''' We calculate gradients with respect to moving each point in x or y direction '''

        self.make_edges()
        print('Calculating gradients for {} edges'.format(len(self.edges)))
        gradient_pairs_edges=[]
        for edge in self.edges:
            gradient_pairs_edges.append(edge.derivative(gradient_fields,wavelength,n_points=self.edge_precision,real=real))
            print('.')
        print('')
        #the gradients returned for an edge derivative are the gradients with respect to moving each end point perpendicular to that edge
        #This is not exactly what we are looking for here, since we want the derivative w/ respect to moving each point
        #in the x or y direction, so coming up is a lot of projections...

        gradients=[]

        for i,point in enumerate(self.points):

            deriv_edge_1=gradient_pairs_edges[i][1]
            normal_edge_1=self.edges[i].normal
            deriv_edge_2=gradient_pairs_edges[(i+1)%len(self.edges)][0]
            normal_edge_2=self.edges[(i+1)%len(self.edges)].normal

            deriv_x=np.dot(deriv_edge_1*normal_edge_1+deriv_edge_2*normal_edge_2,[1,0,0])
            deriv_y=np.dot(deriv_edge_1*normal_edge_1+deriv_edge_2*normal_edge_2,[0,1,0])

            gradients.append(deriv_x)
            gradients.append(deriv_y)

        self.gradients.append(gradients)
        return self.gradients[-1]

    def update_geometry(self,points_linear):
        '''Sets the points. Must be fed a linear array of points, because during the optimization the point coordinates are not by pair'''
        self.points =np.reshape(points_linear,(-1,2))

    def update_geometry_points(self, points):
        '''For debugging. takes in point coordinates as tuples :)'''
        self.points = points


    def get_current_params(self):
        '''returns the points coordinates linearly '''
        return np.reshape(self.points,(-1)).copy()

    def initialize(self,wavelengths,opt):
        self.eps_in.initialize(wavelengths)
        self.eps_out.initialize(wavelengths)
        self.opt=opt

    def add_geo(self, sim, params=None):
        ''' Adds the geometry to a Lumerical simulation'''

        fdtd = sim.fdtd

        if params is None:
            points = self.points
        else:
            points = np.reshape(params, (-1, 2))
        fdtd.putv('vertices', points)

        script = ("addpoly;" +
                  "set('name','polygon_{0}');" +
                  "set('z',{1});" +
                  "set('x',0);" +
                  "set('y',0);" +
                  "set('z span',{2});" +
                  "set('vertices',vertices);" +
                  "{3}").format(self.hash, self.z, self.depth, self.eps_in.set_script())
        fdtd.eval(script)

    def update_geo_in_sim(self,sim,params):
        points = np.reshape(params, (-1, 2))
        sim.fdtd.putv('vertices', points)
        script = ("select('polygon_{0}');" +
                  "set('vertices',vertices);").format(self.hash)
        sim.fdtd.eval(script)

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

def taper_splitter(params):
    '''Just a taper where the paramaters are the y coordinates of the nodes of a cubic spline'''
    points_x=np.concatenate(([-1.01e-6],np.linspace(-1e-6,1e-6,18),[1.01e-6]))
    points_y=np.concatenate(([0.25e-6],params,[0.6e-6]))
    n_interpolation_points=100
    polygon_points_x = np.linspace(min(points_x), max(points_x), n_interpolation_points)
    interpolator = scipy.interpolate.interp1d(points_x, points_y, kind='cubic')
    polygon_points_y = interpolator(polygon_points_x)
    polygon_points_y = np.maximum(0.2e-6, (np.minimum(1e-6, polygon_points_y)))
    polygon_points_up = [(x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
    polygon_points_down = [(x, -y) for x, y in zip(polygon_points_x, polygon_points_y)]
    polygon_points = np.array(polygon_points_up[::-1] + polygon_points_down)
    return polygon_points

class function_defined_Polygon(Polygon):
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

    def __init__(self,func=taper_splitter,initial_params=np.linspace(0.25e-6,0.6e-6,18),bounds=None,z=0,depth=220e-9,eps_out=1.44**2,eps_in=3.44**2,edge_precision=10,dx=0.01e-9):
        #given properties
        self.points=func(initial_params)
        self.func=func
        self.z=z
        self.current_params=initial_params
        self.depth=depth
        self.gradients=[]
        self.edge_precision=edge_precision
        self.bounds=bounds
        self.params_hist=[initial_params]
        if type(eps_out) is Material:
            self.eps_out=eps_out
        else:
            self.eps_out=Material(eps_out)
        if type(eps_in) is Material:
            self.eps_in = eps_in
        else:
            self.eps_in = Material(eps_in)
        self.make_edges()
        self.dx=dx
        self.hash = random.getrandbits(128)


        return

    def update_geometry(self,params):
        self.points=self.func(params)
        self.current_params=params
        self.params_hist.append(params)


    def get_current_params(self):
        return self.current_params

    def calculate_gradients(self, gradient_fields, wavelengths,real=True):

        wavelength=wavelengths[0] #TODO THIS IS NOT DEALING WITH MULTIPLE WAVELENGTHS
        polygon_gradients = np.array(
            super(function_defined_Polygon, self).calculate_gradients(gradient_fields, wavelength,real=real))

        polygon_points_linear = self.func(self.current_params).reshape(-1)
        dx = 1e-11
        gradients = []
        for i, param in enumerate(self.current_params):
            d_params = np.array(self.current_params.copy())
            d_params[i] += dx
            d_polygon_points_linear = self.func(d_params).reshape(-1)
            partial_derivs = (d_polygon_points_linear - polygon_points_linear)/dx
            gradients.append(sum(partial_derivs*polygon_gradients))

        self.gradients.append(gradients)

        return self.gradients[-1]

    def add_poly_script(self,points,only_update=False):
        #vertices_string=format(matlab.double(points.tolist())).replace('],[','];[')
        vertices_string=np.array2string(points,max_line_width=10e10,floatmode='unique',separator=',').replace(',\n',';')
        if not only_update:
            script = (  "addpoly;"
                      + "set('name','polygon_{0}');" ).format(self.hash)
        else:
            script = "select('polygon_{0}');".format(self.hash)

        script += (  "set('z',{1});"
                   + "set('x',0);"
                   + "set('y',0);"
                   + "set('z span',{2});"
                   + "set('vertices',{3});"
                   + "{4}" ).format(self.hash, self.z, self.depth,vertices_string, self.eps_in.set_script())
        return script

    def add_geo(self, sim, params=None,eval=True,only_update=False):
        ''' Adds the geometry to a Lumerical simulation'''
        fdtd = sim.fdtd
        if params is None:
            points = self.points
        else:
            points = self.func(params)

        script=self.add_poly_script(points,only_update)
        if eval:
            fdtd.eval(script)
        return script

    def update_geo_in_sim(self, sim, params, eval=False):
        return self.add_geo(sim,params,eval,only_update=True)

def cross(params):
    '''Example of a function for a cross defined by a spline'''
    y_end = params[-1]
    x_end = 0 - y_end
    points_x = np.concatenate(([-2.01e-6], np.linspace(-2e-6, x_end, 10)))
    points_y = np.concatenate(([0.25e-6], params))
    n_interpolation_points = 50
    polygon_points_x = np.linspace(min(points_x), max(points_x), n_interpolation_points)
    interpolator = scipy.interpolate.interp1d(points_x, points_y, kind='cubic')
    polygon_points_y = [max(min(point, 1e-6), -1e-6) for point in interpolator(polygon_points_x)]

    # pp= polygon points
    # r=right l=left
    # u=up d=down
    pplu = [(x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
    ppld = [(x, -y) for x, y in zip(polygon_points_x, polygon_points_y)]
    ppdl = [(-y, x) for x, y in zip(polygon_points_x, polygon_points_y)]
    ppdr = [(y, x) for x, y in zip(polygon_points_x, polygon_points_y)]
    pprd = [(-x, -y) for x, y in zip(polygon_points_x, polygon_points_y)]
    ppru = [(-x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
    ppur = [(y, -x) for x, y in zip(polygon_points_x, polygon_points_y)]
    ppul = [(-y, -x) for x, y in zip(polygon_points_x, polygon_points_y)]
    polygon_points = np.array(
        pplu[::-1] + ppld[:-1] + ppdl[::-1] + ppdr[:-1] + pprd[::-1] + ppru[:-1] + ppur[::-1] + ppul[:-1])

    return polygon_points
