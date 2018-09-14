from lumopt.utilities.fields import Fields, FieldsNoInterp
import numpy as np

c = 299792458
eps0=8.854187e-12


def get_source_power(fdtd, wl=1550e-9):
    fdtd.eval('sp=sourcepower({});'.format(c/wl))
    source_power = fdtd.getv('sp')
    return source_power

def add_point_monitor(fdtd,monitor_name,position):
    '''This script adds a point monitor in a simulation'''
    script = "addpower;" \
             "set('name','{}');" \
             "set('monitor type','Point');" \
             "set('x',{});" \
             "set('y',{});" \
             "set('z',{});".format(monitor_name,position[0],position[1],position[2])
    fdtd.eval(script)

def add_point_monitor_script(monitor_name,position):
    '''This script adds a point monitor in a simulation'''
    script = "addpower;" \
             "set('name','{}');" \
             "set('monitor type','Point');" \
             "set('x',{});" \
             "set('y',{});" \
             "set('z',{});".format(monitor_name,position[0],position[1],position[2])
    return script

def remove_interpolation_on_monitor(fdtd,monitor_name='opt_fields'):
    ''' Removes the interpolation from the monitor (not to be used lightly!)'''

    script="select({});" \
           "set('spatial interpolation','none');".format(monitor_name)
    fdtd.eval(script)

    return

def get_fields_no_interp(fdtd, monitor_name, get_eps=False, get_D=False, get_H=False):
    '''This function fetches fields from a monitor.
    For the fields used to create gradients, the D component of the field and the refractive index also need to be fetched.
    For the fields on figure of merit monitors, D  or index does not need to be fetched
    It returns a Field object'''
    #print 'getting raw fields for {}'.format(monitor_name)
    # script="m='{}';".format(monitor_name)
    # script+="x = getdata(m,'x');"\
    #         "y = getdata(m,'y');"\
    #         "z = getdata(m,'z');"\
    #         "f = getdata(m,'f');"\
    #         "delta_x = getdata(m,'delta_x');"\
    #         "delta_y = getdata(m,'delta_y');"\
    #         "if (getdata(m,'dimension')==2){ delta_z=0; } else { delta_z=getdata(m,'delta_z');}"\
    #         "Ex = getdata(m, 'Ex');" \
    #         "Ey = getdata(m, 'Ey');"\
    #         "Ez = getdata(m, 'Ez');"
    # lumapi.evalScript(handle, script)
    # fields_x = lumapi.getVar(handle, "x")
    # fields_y = lumapi.getVar(handle, "y")
    # fields_z = lumapi.getVar(handle, "z")
    # fields_Ex = lumapi.getVar(handle,"Ex")
    # fields_Ey = lumapi.getVar(handle,"Ey")
    # fields_Ez = lumapi.getVar(handle,"Ez")
    # fields_E = np.stack((fields_Ex,fields_Ey,fields_Ez),axis=-1)
    # fields_f = lumapi.getVar(handle,"f")
    # delta_x = lumapi.getVar(handle,"delta_x")
    # delta_y = lumapi.getVar(handle,"delta_y")
    # delta_z = lumapi.getVar(handle,"delta_z")
    # deltas=[delta_x,delta_y,delta_z]
    #
    # fields_lambda=c/fields_f
    #
    if get_eps:
        script="index_x=getdata('{0}','index_x');" \
               "index_y=getdata('{0}','index_y');" \
               "index_z=getdata('{0}','index_z');" \
               "eps_x=index_x^2;" \
               "eps_y=index_y^2;" \
               "eps_z=index_z^2;".format(monitor_name+ '_index')
        fdtd.eval(script)
        fields_eps_x = fdtd.getv("eps_x")
        fields_eps_y = fdtd.getv("eps_y")
        fields_eps_z = fdtd.getv("eps_z")
        fields_eps=np.stack((fields_eps_x,fields_eps_y,fields_eps_z),axis=-1)
    else:
        fields_eps = None

    # if get_H:
    #     script = "fields=getresult('{}','H');".format(monitor_name)
    #     lumapi.evalScript(handle, script)
    #     lumapi.evalScript(handle, "fields_H=fields.H;")
    #     fields_H = lumapi.getVar(handle, "fields_H")
    # else:
    #     fields_H = None

    fields = fdtd.getresult(monitor_name,'E')
    delta_x=fdtd.getresult(monitor_name,'delta_x')
    delta_y=fdtd.getresult(monitor_name,'delta_y')
    delta_z = np.array(0) if fdtd.getresult(monitor_name,'dimension')==2. else fdtd.getresult(monitor_name,'delta_z')
    deltas = [delta_x, delta_y, delta_z]
    if get_D:
        fields_D = fields['E']*fields_eps*eps0
    else:
        fields_D = None
    fields_H = fdtd.getresult(monitor_name,'H')['H'] if get_H else None


    return FieldsNoInterp(fields['x'], fields['y'], fields['z'], fields['lambda'], deltas,fields['E'] ,fields_D, fields_eps, fields_H)

def get_fields_interp(fdtd, monitor_name, get_eps=False, get_D=False, get_H=False):
    '''This function fetches fields from a monitor.
    For the fields used to create gradients, the D component of the field and the refractive index also need to be fetched.
    For the fields on figure of merit monitors, D  or index does not need to be fetched
    It returns a Field object'''
    #print 'getting interpolated fields for {}'.format(monitor_name)
    fields = fdtd.getresult(monitor_name,'E')
    fields_eps = fdtd.getresult(monitor_name + '_D_index','eps')['eps'] if get_eps else None
    fields_D = fdtd.getresult(monitor_name + '_D_index','D')['D'] if get_D else None
    fields_H = fdtd.getresult(monitor_name,'H')['H'] if get_H else None

    return Fields(fields['x'], fields['y'], fields['z'], fields['lambda'], fields['E'], fields_D, fields_eps, fields_H)

def get_fields(fdtd, monitor_name, get_eps=False, get_D=False, get_H=False,nointerpolation=False):
    '''This function fetches fields from a monitor.
    For the fields used to create gradients, the D component of the field and the refractive index also need to be fetched.
    For the fields on figure of merit monitors, D  or index does not need to be fetched
    It returns a Field object'''

    if nointerpolation:
        return get_fields_no_interp(fdtd, monitor_name, get_eps, get_D, get_H)
    else:
        return get_fields_interp(fdtd, monitor_name, get_eps, get_D, get_H)

def get_eps_from_sim(fdtd):
    script='select("FDTD");' \
           'set("simulation time",0.1e-15);' \
           'save("eps_extract");' \
           'run;'
    fdtd.eval(script)
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
    fdtd.eval(script)
    fields_eps_x = fdtd.getv("eps_x")
    fields_eps_y = fdtd.getv("eps_y")
    fields_eps_z = fdtd.getv("eps_z")
    x = fdtd.getv("x")
    y = fdtd.getv("y")
    z = fdtd.getv("z")
    fields_eps = np.stack((fields_eps_x, fields_eps_y, fields_eps_z), axis=-1)

    return fields_eps,x,y,z

def set_spatial_interp(fdtd,monitor_name,setting):
    script='select("{}");set("spatial interpolation","{}");'.format(monitor_name,setting)
    fdtd.eval(script)

def get_fields_modesource(fdtd, source_name, get_H=True,direction='Forward'):
    '''This function fetches fields from a mode source monitor.
    It returns a Field object'''

    fields = fdtd.getresult(source_name,'mode profile')

    if not get_H:
        fields['H'] = None
    if direction=='Backward':
        fields['H']=-fields['H']

    return Fields(fields['x'], fields['y'], fields['z'], fields['lambda'], fields['E'], None, None, fields['H'])

def add_imported_source(fdtd,fields,sourcename='imported_source',fommonitorname='fom',wavelengths=[1.55e-6],direction='Forward'):
    fdtd.eval('addimportedsource;')
    fdtd.eval('set("name","{}");'.format(sourcename))
    copy_properties(fdtd, fommonitorname, sourcename, properties=['x', 'y', 'z', 'x_span', 'y_span', 'z_span'])
    make_fields_dataset(fdtd,fields,prefactor=1)
    fdtd.eval('importdataset(EM);')
    #fdtd.eval('set("multifrequency field profile",1);')
    fdtd.eval('set("wavelength start",{});'.format(min(wavelengths)))
    fdtd.eval('set("wavelength stop",{});'.format(max(wavelengths)))
    fdtd.eval('set("direction","{}");'.format(direction))

def make_fields_dataset(fdtd,fields,prefactor):
    fdtd.putv('x',fields.x)
    fdtd.putv('y', fields.y)
    fdtd.putv('z', fields.z)
    fdtd.putv('wavelengths',fields.wl)
    fdtd.putv('H',fields.H)
    fdtd.putv("E",fields.E)
    fdtd.putv('prefactor',[prefactor])
    fdtd.eval('prefactor={};'.format(prefactor))
    fdtd.eval('EM = rectilineardataset("EM fields", x, y, z);')
    fdtd.eval('EM.addparameter("lambda", wavelengths, "f", c/wavelengths);')  # Optional
    fdtd.eval('EM.addattribute("E", E*prefactor);')
    fdtd.eval('EM.addattribute("H", H*prefactor);')  # Optional

def copy_properties(fdtd,origin,destination,properties= ['x', 'y', 'z', 'x_span', 'y_span', 'z_span']):
    orig = fdtd.getObjectById(origin)
    dest = fdtd.getObjectById(destination)
    for thing in properties:
        try:
            dest.__setattr__(thing.replace(' ','_'), orig.__getattr__(thing.replace(' ','_')))
        except:
            print 'Could not copy {} from {} to {} '.format(thing,origin,destination)

def set_injection_axis(fdtd,source_name):
    src = fdtd.getObjectById(source_name)
    if 0==src.z_span: src.injection_axis = 'z-axis'
    if 0==src.y_span: src.injection_axis = 'y-axis'
    if 0==src.x_span: src.injection_axis = 'x-axis'


def add_index_monitors_to_fields_monitors(fdtd, monitor_name):
    '''Adds an index monitor with the same size as a given field monitor'''
    index_monitor_name = monitor_name + '_index'
    fdtd.eval('addindex; set("name","{}");'.format(index_monitor_name))
    things_to_set = ['simulation type', 'monitor type', 'x', 'y', 'z', 'x span', 'y span', 'z span']
    copy_properties(fdtd,monitor_name,index_monitor_name,things_to_set)

    fdtd.eval("set('spatial interpolation','nearest mesh cell');")

    pass


def add_D_monitors_to_fields_monitors(fdtd, monitor_name):
    '''Adds an index monitor with the same size as a given field monitor'''
    index_monitor_name = monitor_name + '_D_index'
    fdtd.eval('addobject("displacement_field_adv"); set("name","{}");'.format(index_monitor_name))
    things_to_set = ['x', 'y', 'z', 'x span', 'y span', 'z span']
    copy_properties(fdtd,monitor_name,index_monitor_name,things_to_set)

    pass

def add_index_to_fields_monitors(fdtd, monitor_name):
    '''Adds an index monitor with the same size as a given field monitor'''
    index_monitor_name = monitor_name + '_index'
    fdtd.eval('addindex; set("name","{}");'.format(index_monitor_name))
    copy_properties(fdtd,monitor_name,index_monitor_name,['monitor type','x', 'y', 'z', 'x span', 'y span', 'z span'])
    fdtd.eval("set('spatial interpolation','None');")



def add_dipole(fdtd, x, y, z, wavelength, dipole, name_suffix='', magnetic=False):

    lumerical_script=add_dipole_script(x, y, z, wavelength, dipole, name_suffix, magnetic)
    fdtd.eval(lumerical_script)


def add_dipole_script(x, y, z, wavelength, dipole, name_suffix='', magnetic=False):
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

def add_magnetic_dipole(x, y, z, wavelength, dipole, name_suffix=''):
    pass

    return
