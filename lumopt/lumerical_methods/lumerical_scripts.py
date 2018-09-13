import lumapi
from lumopt.utilities.fields import Fields, FieldsNoInterp
import numpy as np

c = 299792458
eps0=8.854187e-12


def get_source_power(handle, wl=1550e-9):
    lumapi.evalScript(handle, 'sp=sourcepower({});'.format(c/wl))
    source_power = lumapi.getVar(handle, 'sp')
    return source_power

def add_point_monitor(handle,monitor_name,position):
    '''This script adds a point monitor in a simulation'''
    script = "addpower;" \
             "set('name','{}');" \
             "set('monitor type','Point');" \
             "set('x',{});" \
             "set('y',{});" \
             "set('z',{});".format(monitor_name,position[0],position[1],position[2])
    lumapi.evalScript(handle, script)

def add_point_monitor_script(monitor_name,position):
    '''This script adds a point monitor in a simulation'''
    script = "addpower;" \
             "set('name','{}');" \
             "set('monitor type','Point');" \
             "set('x',{});" \
             "set('y',{});" \
             "set('z',{});".format(monitor_name,position[0],position[1],position[2])
    return script

def remove_interpolation_on_monitor(handle,monitor_name='opt_fields'):
    ''' Removes the interpolation from the monitor (not to be used lightly!)'''

    script="select({});" \
           "set('spatial interpolation','none');".format(monitor_name)
    lumapi.evalScript(handle,script)

    return

def get_fields_no_interp(handle, monitor_name, get_eps=False, get_D=False, get_H=False):
    '''This function fetches fields from a monitor.
    For the fields used to create gradients, the D component of the field and the refractive index also need to be fetched.
    For the fields on figure of merit monitors, D  or index does not need to be fetched
    It returns a Field object'''
    #print 'getting raw fields for {}'.format(monitor_name)
    script="m='{}';".format(monitor_name)
    script+="x = getdata(m,'x');"\
            "y = getdata(m,'y');"\
            "z = getdata(m,'z');"\
            "f = getdata(m,'f');"\
            "delta_x = getdata(m,'delta_x');"\
            "delta_y = getdata(m,'delta_y');"\
            "if (getdata(m,'dimension')==2){ delta_z=0; } else { delta_z=getdata(m,'delta_z');}"\
            "Ex = getdata(m, 'Ex');" \
            "Ey = getdata(m, 'Ey');"\
            "Ez = getdata(m, 'Ez');"
    lumapi.evalScript(handle, script)
    fields_x = lumapi.getVar(handle, "x")
    fields_y = lumapi.getVar(handle, "y")
    fields_z = lumapi.getVar(handle, "z")
    fields_Ex = lumapi.getVar(handle,"Ex")
    fields_Ey = lumapi.getVar(handle,"Ey")
    fields_Ez = lumapi.getVar(handle,"Ez")
    fields_E = np.stack((fields_Ex,fields_Ey,fields_Ez),axis=-1)
    fields_f = lumapi.getVar(handle,"f")
    delta_x = lumapi.getVar(handle,"delta_x")
    delta_y = lumapi.getVar(handle,"delta_y")
    delta_z = lumapi.getVar(handle,"delta_z")
    deltas=[delta_x,delta_y,delta_z]

    fields_lambda=c/fields_f

    if get_eps:
        script="index_x=getdata('{0}','index_x');" \
               "index_y=getdata('{0}','index_y');" \
               "index_z=getdata('{0}','index_z');" \
               "eps_x=index_x^2;" \
               "eps_y=index_y^2;" \
               "eps_z=index_z^2;".format(monitor_name+ '_index')
        lumapi.evalScript(handle,script)
        fields_eps_x = lumapi.getVar(handle, "eps_x")
        fields_eps_y = lumapi.getVar(handle, "eps_y")
        fields_eps_z = lumapi.getVar(handle, "eps_z")
        fields_eps=np.stack((fields_eps_x,fields_eps_y,fields_eps_z),axis=-1)
    else:
        fields_eps = None
    if get_D:
        fields_D=fields_E*fields_eps*eps0

    else:
        fields_D = None
    if get_H:
        script = "fields=getresult('{}','H');".format(monitor_name)
        lumapi.evalScript(handle, script)
        lumapi.evalScript(handle, "fields_H=fields.H;")
        fields_H = lumapi.getVar(handle, "fields_H")
    else:
        fields_H = None

    return FieldsNoInterp(fields_x, fields_y, fields_z, fields_lambda, deltas,fields_E ,fields_D, fields_eps, fields_H)



def get_fields_interp(handle, monitor_name, get_eps=False, get_D=False, get_H=False):
    '''This function fetches fields from a monitor.
    For the fields used to create gradients, the D component of the field and the refractive index also need to be fetched.
    For the fields on figure of merit monitors, D  or index does not need to be fetched
    It returns a Field object'''
    #print 'getting interpolated fields for {}'.format(monitor_name)
    script = "fields=getresult('{}','E');".format(monitor_name)

    lumapi.evalScript(handle, script)
    lumapi.evalScript(handle, "fields_x=fields.x;")
    fields_x = lumapi.getVar(handle, "fields_x")
    lumapi.evalScript(handle, "fields_y=fields.y;")
    fields_y = lumapi.getVar(handle, "fields_y")
    lumapi.evalScript(handle, "fields_z=fields.z;")
    fields_z = lumapi.getVar(handle, "fields_z")
    lumapi.evalScript(handle, "fields_lambda=fields.lambda;")
    fields_lambda = lumapi.getVar(handle, "fields_lambda")
    lumapi.evalScript(handle, "fields_E=fields.E;")
    fields_E = lumapi.getVar(handle, "fields_E")
    if get_eps:
        lumapi.evalScript(handle, "eps=getresult('{}','eps');".format(monitor_name + '_D_index'))
        lumapi.evalScript(handle, "fields_eps=eps.eps;")
        fields_eps = lumapi.getVar(handle, "fields_eps")  # this should have all the field directions
    else:
        fields_eps = None
    if get_D:
        script = "fields=getresult('{}','D');".format(monitor_name + '_D_index')
        lumapi.evalScript(handle, script)
        lumapi.evalScript(handle, "fields_D=fields.D;")
        fields_D = lumapi.getVar(handle, "fields_D")
    else:
        fields_D = None
    if get_H:
        script = "fields=getresult('{}','H');".format(monitor_name)
        lumapi.evalScript(handle, script)
        lumapi.evalScript(handle, "fields_H=fields.H;")
        fields_H = lumapi.getVar(handle, "fields_H")
    else:
        fields_H = None

    return Fields(fields_x, fields_y, fields_z, fields_lambda, fields_E, fields_D, fields_eps, fields_H)


def get_fields(handle, monitor_name, get_eps=False, get_D=False, get_H=False,nointerpolation=False):
    '''This function fetches fields from a monitor.
    For the fields used to create gradients, the D component of the field and the refractive index also need to be fetched.
    For the fields on figure of merit monitors, D  or index does not need to be fetched
    It returns a Field object'''

    if nointerpolation:

        return get_fields_no_interp(handle, monitor_name, get_eps, get_D, get_H)
    else:
        return get_fields_interp(handle, monitor_name, get_eps, get_D, get_H)

def get_eps_from_sim(handle):
    script='select("FDTD");' \
           'set("simulation time",0.1e-15);' \
           'save("eps_extract");' \
           'run;'
    lumapi.evalScript(handle,script)
    # lumapi.evalScript(handle, "eps=getresult('{}','eps');".format('opt_fields' + '_index'))
    # #lumapi.evalScript(handle, "eps=getresult('{}','eps');".format('opt_fields' + '_D_index'))
    # lumapi.evalScript(handle, "fields_eps=eps.eps;")
    # lumapi.evalScript(handle, "x=eps.x;")
    # lumapi.evalScript(handle, "y=eps.y;")
    # lumapi.evalScript(handle, "z=eps.z;")
    # fields_eps=lumapi.getVar(handle,'fields_eps')
    # x=lumapi.getVar(handle,'x')
    # y=lumapi.getVar(handle,'y')
    # z=lumapi.getVar(handle,'z')

    script = "index_x=getdata('{0}','index_x');" \
             "index_y=getdata('{0}','index_y');" \
             "index_z=getdata('{0}','index_z');" \
             "eps_x=index_x^2;" \
             "eps_y=index_y^2;" \
             "eps_z=index_z^2;" \
             "x = getdata('{0}','x');"\
             "y = getdata('{0}','y');"\
             "z = getdata('{0}','z');".format('opt_fields' + '_index')
    lumapi.evalScript(handle, script)
    fields_eps_x = lumapi.getVar(handle, "eps_x")
    fields_eps_y = lumapi.getVar(handle, "eps_y")
    fields_eps_z = lumapi.getVar(handle, "eps_z")
    x = lumapi.getVar(handle, "x")
    y = lumapi.getVar(handle, "y")
    z = lumapi.getVar(handle, "z")
    fields_eps = np.stack((fields_eps_x, fields_eps_y, fields_eps_z), axis=-1)

    #lumapi.close(handle)
    return fields_eps,x,y,z

def set_spatial_interp(handle,monitor_name,setting):
    script='select("{}");set("spatial interpolation","{}");'.format(monitor_name,setting)
    lumapi.evalScript(handle,script)

def get_fields_modesource(handle, source_name, get_H=True,direction='Forward'):
    '''This function fetches fields from a mode source monitor.
    It returns a Field object'''

    script = "fields=getresult('{}','mode profile');".format(source_name)
    lumapi.evalScript(handle, script)
    lumapi.evalScript(handle, "fields_x=fields.x;")
    fields_x = lumapi.getVar(handle, "fields_x")
    lumapi.evalScript(handle, "fields_y=fields.y;")
    fields_y = lumapi.getVar(handle, "fields_y")
    lumapi.evalScript(handle, "fields_z=fields.z;")
    fields_z = lumapi.getVar(handle, "fields_z")
    lumapi.evalScript(handle, "fields_lambda=fields.lambda;")
    fields_lambda = lumapi.getVar(handle, "fields_lambda")
    lumapi.evalScript(handle, "fields_E=fields.E;")
    fields_E = lumapi.getVar(handle, "fields_E")

    if get_H:
        lumapi.evalScript(handle, "fields_H=fields.H;")
        fields_H = lumapi.getVar(handle, "fields_H")
    else:
        fields_H = None
    if direction=='Backward':
        fields_H=-fields_H

    return Fields(fields_x, fields_y, fields_z, fields_lambda, fields_E, None, None, fields_H)

def add_imported_source(handle,fields,sourcename='imported_source',fommonitorname='fom',wavelengths=[1.55e-6],direction='Forward'):
    lumapi.evalScript(handle,'addimportedsource;')
    lumapi.evalScript(handle,'set("name","{}");'.format(sourcename))
    copy_properties(handle, fommonitorname, sourcename, properties=['x', 'y', 'z', 'x span', 'y span', 'z span'])
    make_fields_dataset(handle,fields,prefactor=1)
    lumapi.evalScript(handle,'importdataset(EM);')
    #lumapi.evalScript(handle,'set("multifrequency field profile",1);')
    lumapi.evalScript(handle,'set("wavelength start",{});'.format(min(wavelengths)))
    lumapi.evalScript(handle,'set("wavelength stop",{});'.format(max(wavelengths)))
    lumapi.evalScript(handle,'set("direction","{}");'.format(direction))


def make_fields_dataset(handle,fields,prefactor):
    lumapi.putMatrix(handle,'x',fields.x)
    lumapi.putMatrix(handle, 'y', fields.y)
    lumapi.putMatrix(handle, 'z', fields.z)
    lumapi.putMatrix(handle,'wavelengths',fields.wl)
    lumapi.putMatrix(handle,'H',fields.H)
    lumapi.putMatrix(handle,"E",fields.E)
    lumapi.putMatrix(handle,'prefactor',[prefactor])
    lumapi.evalScript(handle,'prefactor={};'.format(prefactor))
    lumapi.evalScript(handle,'EM = rectilineardataset("EM fields", x, y, z);')
    lumapi.evalScript(handle,'EM.addparameter("lambda", wavelengths, "f", c/wavelengths);')  # Optional
    lumapi.evalScript(handle,'EM.addattribute("E", E*prefactor);')
    lumapi.evalScript(handle,'EM.addattribute("H", H*prefactor);')  # Optional

def copy_properties(handle,origin,destination,properties= ['x', 'y', 'z', 'x span', 'y span', 'z span']):
    for thing in properties:
        script = "select('{0}');temp=get('{1}');select('{2}');set('{1}',temp);".format(origin, thing,
                                                                                       destination)
        try:
            lumapi.evalScript(handle, script)
        except:
            print 'Could not copy {} from {} to {} '.format(thing,origin,destination)

def set_injection_axis(handle,source_name):
    script="select('{}');".format(source_name)
    script=script+"if(0==get('z span')){set('injection axis','z axis');}" \
           "if(0==get('y span')){set('injection axis','y-axis');}" \
           "if(0==get('x span')){set('injection axis','x-axis');}"

    lumapi.evalScript(handle,script)



def add_index_monitors_to_fields_monitors(handle, monitor_name):
    '''Adds an index monitor with the same size as a given field monitor'''
    index_monitor_name = monitor_name + '_index'
    lumapi.evalScript(handle, 'addindex; set("name","{}");'.format(index_monitor_name))
    things_to_set = ['simulation type', 'monitor type', 'x', 'y', 'z', 'x span', 'y span', 'z span']
    for thing in things_to_set:
        script = 'select("{0}");temp=get("{1}");select("{2}");set("{1}",temp);'.format(monitor_name, thing,
                                                                                       index_monitor_name)
        try:
            lumapi.evalScript(handle, script)
        except:
            print script
    lumapi.evalScript(handle, "set('spatial interpolation','nearest mesh cell');")

    pass


def add_D_monitors_to_fields_monitors(handle, monitor_name):
    '''Adds an index monitor with the same size as a given field monitor'''
    index_monitor_name = monitor_name + '_D_index'
    lumapi.evalScript(handle, 'addobject("displacement_field_adv"); set("name","{}");'.format(index_monitor_name))
    things_to_set = ['x', 'y', 'z', 'x span', 'y span', 'z span']
    for thing in things_to_set:
        script = "select('{0}');temp=get('{1}');select('{2}');set('{1}',temp);".format(monitor_name, thing,
                                                                                       index_monitor_name)
        try:
            lumapi.evalScript(handle, script)
        except:
            print script

    pass

def add_index_to_fields_monitors(handle, monitor_name):
    '''Adds an index monitor with the same size as a given field monitor'''
    index_monitor_name = monitor_name + '_index'
    lumapi.evalScript(handle, 'addindex; set("name","{}");'.format(index_monitor_name))
    copy_properties(handle,monitor_name,index_monitor_name,['monitor type','x', 'y', 'z', 'x span', 'y span', 'z span'])
    lumapi.evalScript(handle, "set('spatial interpolation','None');")


def add_dipole(handle, x, y, z, wavelength, dipole, name_suffix='', magnetic=False):

    lumerical_script=add_dipole_script(handle, x, y, z, wavelength, dipole, name_suffix, magnetic)
    lumapi.evalScript(handle, lumerical_script)


def add_dipole_script(handle, x, y, z, wavelength, dipole, name_suffix='', magnetic=False):
    script=''
    for dir, phasor in zip(['x', 'y', 'z'], dipole):

        theta = 0
        phi = 0
        if dir == 'x':
            theta = 90
        if dir == 'y':
            theta = 90
            phi = 90

        phase_deg = np.angle(phasor)*180/np.pi
        amplitude = np.abs(phasor)
        if not amplitude==0:
            lumerical_script = "adddipole;"
            if magnetic: lumerical_script += "set('dipole type','Magnetic dipole');"
            lumerical_script += "set('name','{8}');" \
                                "set('x',{0});" \
                                "set('y',{1});" \
                                "set('z',{2});" \
                                "set('theta',{3});" \
                                "set('phi',{4});" \
                                "set('phase',{5});" \
                                "set('center wavelength',{6});" \
                                "set ('wavelength span',0);" \
                                "baseAmp=get('base amplitude');" \
                                "set('amplitude',{7}/baseAmp);".format(x, y, z, theta, phi, phase_deg, wavelength,
                                                                       amplitude, dir + name_suffix)

            script+=lumerical_script

    return script

def add_magnetic_dipole(handle, x, y, z, wavelength, dipole, name_suffix=''):
    pass

    return
