""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

from scipy.interpolate import RegularGridInterpolator
import numpy as np


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
