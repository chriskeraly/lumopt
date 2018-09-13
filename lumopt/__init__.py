import inspect, os
import pathlib

here=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
here=pathlib.Path(here)
root=pathlib.Path(*here.parts[:-1])
CONFIG={}
#CONFIG['fdtd_path']="/Applications/Lumerical/FDTD Solutions/FDTD Solutions.app/Contents/API/Python"
CONFIG['root']=str(root.absolute())
print 'CONFIGURATION FILE {}'.format(CONFIG)

## Geeze this seems awefully complicated, is there no other way to do it better??
