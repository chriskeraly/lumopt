import os, sys, platform

# add lumopt.py to system path
lumopt_dir_name = os.path.dirname(__file__)
parent_dir_name = os.path.dirname(lumopt_dir_name)
if parent_dir_name not in sys.path:
    sys.path.append(parent_dir_name)

# look for lumapi.py in system path
python_api_path = ''
for dir_name in sys.path:
    if os.path.isfile(os.path.join(dir_name, 'lumapi.py')):
        python_api_path = dir_name; break
# if search comes out empty, look in the default install path
if not python_api_path:
    current_platform = platform.system()
    default_api_path = ''
    if current_platform == 'Windows':
        default_api_path = '/Program Files/Lumerical/2019b/api/python'
    elif current_platform == 'Darwin':
        default_api_path = '/Applications/Lumerical/FDTD/FDTD.app/Contents/MacOS/'
    elif current_platform == 'Linux':
        default_api_path = '/opt/lumerical/fdtd/api/python'
    default_api_path = os.path.normpath(default_api_path)
    if os.path.isfile(os.path.join(default_api_path, 'lumapi.py')):
        sys.path.append(default_api_path)
        python_api_path = default_api_path

# save paths for subsequent file access
CONFIG = {'root' : parent_dir_name, 'lumapi' : python_api_path}
print('CONFIGURATION FILE {}'.format(CONFIG))
