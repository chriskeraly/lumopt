from scipy.interpolate import RegularGridInterpolator
import numpy as np

from scipy.integrate import simps
import numpy as np

def trapz1D(vals,x):
    '''This is trapz but that is ok with one dimensional values'''
    if len(x)==1:
        try:
            return vals[0]
        except:
            return vals
    else:
        return np.trapz(vals,x)

def trapz2D(vals,x,y):
    '''2D trapz methods, which is OK with one dimensional values'''

    temp=[]
    for i in range(len(y)):
        temp.append(trapz1D(vals[:,i],x))

    return trapz1D(temp,y)


def trapz3D( vals,x, y,z,):
    '''3D trapz methods, which is OK with one dimensional values'''

    def process_input(input):
        if type(input) is float:
            input = np.array([input])
        else:
            input = input.squeeze()
        if input.shape == ():
            input = np.array([input])
        return input

    x, y, z = map(process_input, [x, y, z])

    temp = []
    for i in range(len(z)):
        temp.append(trapz2D(vals[:,:,i],x,y))

    return trapz1D(temp, z)


def dblsimps(func,x_min,x_max,y_min,y_max,points_per_dimension=1):
    '''Integrates a function over a square, using only a few sample points'''
    if points_per_dimension==1:
        return (x_max-x_min)*(y_max-y_min)*func((x_max+x_min)/2,(y_max+y_min)/2)
    else:
        x_array = np.linspace(x_min, x_max, points_per_dimension)
        y_array = np.linspace(y_min, y_max, points_per_dimension)
    z=np.array([func(x,y) for x in x_array for y in y_array])
    z=z.reshape(points_per_dimension,points_per_dimension)

    return simps(simps(z, y_array), x_array)


def wrapped_GridInterpolator(points, values, method='linear', bounds_error=True, fill_value=float('nan')):
    '''This is a wrapper around Scipy's RegularGridInterpolator so that it can deal with entries of 1 dimension

    Original doc:

    The data must be defined on a regular grid; the grid spacing however may be
    uneven.  Linear and nearest-neighbour interpolation are supported. After
    setting up the interpolator object, the interpolation method (*linear* or
    *nearest*) may be chosen at each evaluation.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.

    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.

    method : str, optional
        The method of interpolation to perform. Supported are "linear" and
        "nearest". This parameter will become the default for the object's
        ``__call__`` method. Default is "linear".

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.

    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.

    '''


    dim_1_inputs=[array.size==1 for array in points] #find all 1 dimensional values
    non_dim_1_inputs = [array.size != 1 for array in points]

    newpoints=[]        #points without 1 dimensional entries
    for array in points:
        if array.size>1:
            newpoints.append(array)

    singleton= newpoints==[]

    newvalues=values.copy().squeeze()  #remove all one dimensional entries

    if not singleton:
        interpolator=RegularGridInterpolator(points=tuple(newpoints),values=newvalues,method=method,bounds_error=bounds_error,fill_value=fill_value)

        def wrapped_interpolator(points):
            try:
                newpoints=[]
                for point in points:
                    newpoint = []
                    for x,single_dim in zip(point,dim_1_inputs):
                        if not single_dim:
                            if type(x) is np.ndarray or type(x) is list:
                                newpoint.append(x[0])
                            else:
                                newpoint.append(x) #TODO: make sure this is never used
                    newpoints.append(tuple(newpoint))
                return interpolator(np.array(newpoints))
            except:
                newpoint = []
                for x, single_dim in zip(points, dim_1_inputs):
                    if not single_dim:
                        if type(x) is np.ndarray or type(x) is list:
                            newpoint.append(x)
                        else:
                            newpoint.append(np.array([x]))  # TODO: make sure this is never used

                return interpolator(tuple(newpoint))

    else:
        def wrapped_interpolator(point):
            return newvalues

    return wrapped_interpolator

def demo_interpolator():
    points = (np.array([3, 4]), np.array([1, 2]),np.array([5]))
    vals = np.array([10,11,12,13]).reshape(2,2,1)

    print(vals.shape)
    wi=wrapped_GridInterpolator(points,vals)

    print(wi((3.5,1.5,5)))
    print(wi((3.5, 1, 5)))
    print(wi([(3.5,1.5,5),(3.5,1,5)]))

    points=(np.array([1]),np.array([2]))
    vals=np.array([5]).reshape(1,1)

    wi = wrapped_GridInterpolator(points, vals)

    print(wi((1, 2)))

    ti=RegularGridInterpolator(points,vals)
    try:
        print(ti((3.5,1.5,5)))
    except:
        print('It didnt work for the regular interpolator!')

if __name__=='__main__':
    demo_interpolator()
    import os
    print(os.getcwd())
    print(dblsimps(lambda x, y:x * y, 0, 10, -2, 2,points_per_dimension=3))
