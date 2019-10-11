import lumopt.lumerical_methods.lumerical_scripts as ls
import numpy as np
from scipy import integrate
import lumapi

from lumopt.utilities.wavelengths import Wavelengths

class FieldIntensity(object):
	'''
	A figure of merit which is simply the average |E|^2 in a monitor.

	Original code provided by Christophere Lalau-Keraly. Adapted to the latest version of LumOpt.
	'''

	def __init__(self, monitor_name,wavelengths,subspace='xyz',dipole_inc=1):
		'''
		:param monitor_name: A string: the name of the point monitor
		:param wavelengths: A list of the wavelengths of interest (for the moment supports only a single value)
		:param subspace: A string giving the subspace in which to optimize intensity. 'None' means absolute intensity, 'xy' intensity in xy-plane, 'x' intensity in x-axis etc.
		:param dipole_inc: Include ever dipole_inc dipoles on the monitor. E.g. dipole_inc=2 places dipole at every other position on monitor.
		'''
		self.monitor_name = monitor_name
		self.wavelengths = wavelengths
		self.multi_freq_src=False
		self.subspace=subspace
		self.dipole_inc=dipole_inc

	def initialize(self,sim):
		self.add_adjoint_sources(sim)

	def make_forward_sim(self, sim):

		for src in self.adjoint_source_names:
			sim.fdtd.setnamed(src,'enabled',False)

	def make_adjoint_sim(self,sim):

		FieldIntensity.set_dipoles_on_monitor(sim,self.forward_field,self.adjoint_source_names)

		return


	def get_fom(self, sim):
		'''
		:param simulation: The simulation object of the base simulation
		:return: The figure of merit
		'''
		self.forward_field = sim.fdtd.getresult(self.monitor_name,'E')
		self.wavelengths = FieldIntensity.get_wavelengths(sim)
		
		# Calculate the average E field intensity on monitor
		x = self.forward_field['x'].squeeze()
		y = self.forward_field['y'].squeeze()
		z = self.forward_field['z'].squeeze()

		shape=self.forward_field['E'].shape
		numwls=shape[3]
		E2 = np.empty(shape[:4])

		for l in range(numwls):

			E=self.forward_field['E']

			E2x=('x' in self.subspace)*np.conj(E[:,:,:,l,0])*E[:,:,:,l,0]
			E2y=('y' in self.subspace)*np.conj(E[:,:,:,l,1])*E[:,:,:,l,1]
			E2z=('z' in self.subspace)*np.conj(E[:,:,:,l,2])*E[:,:,:,l,2]

			E2[:,:,:,l] = np.real(	E2x + E2y + E2z )


		E2mean=np.empty(numwls)

		if x.size==1 and y.size==1 and z.size==1:
			return E2.squeeze()	
		if x.size>1 and y.size==1 and z.size==1:
			E2mean = integrate.trapz(E2.squeeze(),x=x,axis=0) / np.abs(x[0]-x[-1])
		if y.size>1 and x.size==1 and z.size==1:
			E2mean = integrate.trapz(E2.squeeze(),x=y,axis=0) / np.abs(y[0]-y[-1])
		if z.size>1 and x.size==1 and y.size==1:
			E2mean = integrate.trapz(E2.squeeze(),x=z,axis=0) / np.abs(z[0]-z[-1])
		if x.size>1 and y.size>1 and z.size==1:
			E2meanX = integrate.trapz(E2.squeeze(),x=x,axis=0) / np.abs(x[0]-x[-1])
			E2mean  = integrate.trapz(E2meanX,x=y,axis=0) / np.abs(y[0]-y[-1])

		fom = E2mean.squeeze()

		return fom

	def add_adjoint_sources(self, sim):
		'''
		Adds the adjoint sources required in the adjoint simulation
		:param simulation: The simulation object of the base simulation
		'''
		monitor_name=self.monitor_name

		# Choose orientations to omit
		oris='xyz'
		omit=oris.replace(self.subspace,'') # only dirs outside subspace retained for omission

		self.adjoint_source_names=FieldIntensity.add_dipoles_on_monitor(sim,monitor_name,omit_ori=omit,dipole_inc=self.dipole_inc)

		return
		
	def get_adjoint_field_scaling(self,sim):
		return np.array([1])

	def fom_gradient_wavelength_integral(self, E_partial_derivs_vs_wl, wl):
		assert np.allclose(wl, self.wavelengths)
		return FieldIntensity.fom_gradient_wavelength_integral_impl(E_partial_derivs_vs_wl, self.wavelengths)

	@staticmethod
	def fom_gradient_wavelength_integral_impl(E_partial_derivs_vs_wl, wl):

		if wl.size > 1:
			assert T_fwd_partial_derivs_vs_wl.shape[1] == wl.size
			
			wavelength_range = wl.max() - wl.min()
			T_fwd_error = T_fwd_vs_wavelength - target_T_fwd_vs_wavelength
			T_fwd_error_integrand = np.power(np.abs(T_fwd_error), norm_p) / wavelength_range
			const_factor = -1.0 * np.power(np.trapz(y = T_fwd_error_integrand, x = wl), 1.0 / norm_p - 1.0)
			integral_kernel = np.power(np.abs(T_fwd_error), norm_p - 1) * np.sign(T_fwd_error) / wavelength_range
			
			## Implement the trapezoidal integration as a matrix-vector-product for performance reasons
			d = np.diff(wl)
			quad_weight = np.append(np.append(d[0], d[0:-1]+d[1:]),d[-1])/2 #< There is probably a more elegant way to do this
			v = const_factor * integral_kernel * quad_weight
			T_fwd_partial_derivs = T_fwd_partial_derivs_vs_wl.dot(v)

			## This is the much slower (but possibly more readable) code
			# num_opt_param = T_fwd_partial_derivs_vs_wl.shape[0]
			# T_fwd_partial_derivs = np.zeros(num_opt_param, dtype = 'complex')
			# for i in range(num_opt_param):
			#     T_fwd_partial_deriv = np.take(T_fwd_partial_derivs_vs_wl.transpose(), indices = i, axis = 1)
			#     T_fwd_partial_derivs[i] = const_factor * np.trapz(y = integral_kernel * T_fwd_partial_deriv, x = wl)
		else:
			E_fwd_partial_derivs = E_partial_derivs_vs_wl.flatten()

		return E_fwd_partial_derivs.flatten().real

	@staticmethod
	def add_dipoles_on_monitor(sim,monitor_name,omit_ori=None,dipole_inc=1):
		'''
		Adds 3 dipoles (on in each direction) to every point on a monitor.
		Returns their names.
		:param monitor_name: string, monitor on which to place dipoles
		:param omit_ori: orientations to omit.

		'''

		# Possible dipole positions, i.e. all mesh positions
		mesh_x=sim.fdtd.getresult('FDTD','x').squeeze()
		mesh_y=sim.fdtd.getresult('FDTD','y').squeeze()
		mesh_z=np.array([sim.fdtd.getresult('FDTD','z')]) # If simulation 2D, make 0 return into an array

		# Monitor positions
		sim.fdtd.select(monitor_name)
		mtype=sim.fdtd.get('monitor type')

		if mtype=='Point':
			dipole_x=np.array([sim.fdtd.get('x')])
			dipole_y=np.array([sim.fdtd.get('y')])
			dipole_z=np.array([sim.fdtd.get('z')])

		if mtype=='Linear X':
			monitor_xmin=sim.fdtd.get('x min')
			monitor_xmax=sim.fdtd.get('x max')
			monitor_y=sim.fdtd.get('y')
			monitor_z=sim.fdtd.get('z')

			greater,lesser=mesh_x>monitor_xmin,mesh_x<monitor_xmax

			dipole_x=mesh_x[greater & lesser]
			dipole_y=[mesh_y[np.argmin(np.abs(monitor_y-mesh_y))]]
			dipole_z=[mesh_z[np.argmin(np.abs(monitor_z-mesh_z))]]

		if mtype=='2D Z-normal':
			monitor_xmin=sim.fdtd.get('x min')
			monitor_xmax=sim.fdtd.get('x max')			
			monitor_ymin=sim.fdtd.get('y min')
			monitor_ymax=sim.fdtd.get('y max')
			monitor_z=sim.fdtd.get('z')

			greater,lesser=mesh_x>monitor_xmin,mesh_x<monitor_xmax
			dipole_x=mesh_x[greater & lesser]		

			greater,lesser=mesh_y>monitor_ymin,mesh_y<monitor_ymax
			dipole_y=mesh_y[greater & lesser]
			dipole_z=[mesh_z[np.argmin(np.abs(monitor_z-mesh_z))]]

		# Include every dipole_inc positions
		dipole_x=dipole_x[::dipole_inc]
		dipole_y=dipole_y[::dipole_inc]
		dipole_z=dipole_z[::dipole_inc]

		# Dipole positions have been defined
		# Now iterate through them
		def dipole_name(mn,i,j,k,ori):
			return 'dipole_src_'+str(i)+'_'+str(j)+'_'+str(k)+'_'+ori

		dipole_sources=[]

		for i in range(len(dipole_x)):
			x=dipole_x[i]
			for j in range(len(dipole_y)):
				y=dipole_y[j]
				for k in range(len(dipole_z)):
					z=dipole_z[k]
					for ori in ['x','y','z']:
						if ori in omit_ori: continue
						pos=[x,y,z]
						dpn=dipole_name(monitor_name,i,j,k,ori)
						dipole_sources.append(dpn)
						FieldIntensity.add_dipole_source(sim,pos,dpn,ori)

		return dipole_sources


	@staticmethod
	def set_dipoles_on_monitor(sim,monitor_E_field,dipole_names):

		Ec=np.conj(monitor_E_field['E'])
		xE=monitor_E_field['x']
		yE=monitor_E_field['y']
		zE=monitor_E_field['z']

		for src in dipole_names:
			sim.fdtd.select(src)
			sim.fdtd.set('enabled',True)

			x=sim.fdtd.get('x');y=sim.fdtd.get('y');z=sim.fdtd.get('z');

			# Maybe not the most robust way to get the proper E-field position
			xarg=np.argmin(np.abs(x-xE))
			yarg=np.argmin(np.abs(y-yE))
			zarg=np.argmin(np.abs(z-zE))

			src_split = src.split('_')
			ori=src_split[5]

			if ori=='x': l=0
			if ori=='y': l=1
			if ori=='z': l=2

			Edip=Ec[xarg,yarg,zarg,0,l]
			amplitude=np.abs(Edip)
			phase= np.angle(Edip,deg=True)

			sim.fdtd.set('amplitude',amplitude)
			sim.fdtd.set('phase',phase)



	@staticmethod
	def add_dipole_source(sim,pos,source_name,orientation):
		'''
		
		orientation: 'x','y',or 'z'
		'''

		sim.fdtd.adddipole()
		src=source_name
		sim.fdtd.set('name',src)
		sim.fdtd.setnamed(src,'x',pos[0])
		sim.fdtd.setnamed(src,'y',pos[1])
		sim.fdtd.setnamed(src,'z',pos[2])

		if(orientation=='x'):
			sim.fdtd.setnamed(src,'theta',90)
		if(orientation=='y'):
			sim.fdtd.setnamed(src,'phi',90)
			sim.fdtd.setnamed(src,'theta',90)

	@staticmethod
	def get_monitor_orientation(sim,monitor_name):

		mtype=sim.fdtd.getnamed(monitor_name,'monitor_type')

		if mtype=='Point':
			return 'Point'
		if mtype=='Linear X' or mtype=='2D X-normal':
			return 'x'
		if mtype=='Linear Y' or mtype=='2D Y-normal':
			return 'y'
		if mtype=='Linear Z' or mtype=='2D Z-normal':
			return 'z'
		else:
			return mtype

	def check_monitor_alignment(self, sim):
	  
		## Here, we check that the FOM_monitor is properly aligned with the mesh
		if sim.fdtd.getnamednumber(self.monitor_name) != 1:
			raise UserWarning('monitor could not be found or the specified name is not unique.')
		
		# Get the orientation
		monitor_type = sim.fdtd.getnamed(self.monitor_name, 'monitor type')

		if (monitor_type == 'Linear X') or (monitor_type == '2D X-normal'):
			orientation = 'x'
		elif (monitor_type == 'Linear Y') or (monitor_type == '2D Y-normal'):
			orientation = 'y'
		elif (monitor_type == 'Linear Z') or (monitor_type == '2D Z-normal'):
			orientation = 'z'
		else:
			raise UserWarning('monitor should be 2D or linear for a mode expansion to be meaningful.')

		monitor_pos = sim.fdtd.getnamed(self.monitor_name, orientation)
		if sim.fdtd.getnamednumber('FDTD') == 1:
			grid = sim.fdtd.getresult('FDTD', orientation)
		elif sim.fdtd.getnamednumber('varFDTD') == 1:
			grid = sim.fdtd.getresult('varFDTD', orientation)
		else:
			raise UserWarning('no FDTD or varFDTD solver object could be found.')
		## Check if this is exactly aligned with the simulation mesh. It exactly aligns if we find a point
		## along the grid which is no more than 'tol' away from the position
		tol = 1e-9
		if min(abs(grid-monitor_pos)) > tol:
			print('WARNING: The monitor "{}" is not aligned with the grid. This can introduce small phase errors which sometimes result in inaccurate gradients.'.format(self.monitor_name))

	@staticmethod
	def get_wavelengths(sim):
		return Wavelengths(sim.fdtd.getglobalsource('wavelength start'), 
						   sim.fdtd.getglobalsource('wavelength stop'),
						   sim.fdtd.getglobalmonitor('frequency points')).asarray()


class FieldIntensities(object):
	'''
	A slightly more complex figure of merit than FieldIntensity.
	Now fields at different points in space can be combined linearly
	to create a figure of merit of the form:

		FOM= |a1*E1+ a2*E2 + ... + an*En|^2

	where ai can be a complex vector of length 3.
	With the right amplitudes ai, this can be used to form a wide
	variety of complex meaningful figures of merit (absorption, modematching).

	If the amplitudes are set to None, then
	FOM=|E1|^2+...+|En|^2
	'''

	def __init__(self, monitor_names, weight_amplitudes=[1, 1], wavelengths=[1550e-9], normalize_to_source_power=False):
		'''
		:param positions: A list of tuples representing the 3D coordinate in space of where the fields should be measured
		:param weight_amplitudes: A list of complex amplitudes
		:param wavelengths: The wavelengths of interest (for the moment supports only one wavelength)
		:param normalize_to_source_power: Should everything be normalized to the source power?
		'''
		self.positions=positions
		self.weight_amplitudes=weight_amplitudes
		self.wavelengths=wavelengths
		self.current_fom=None
		self.fields=None
		self.normalize_to_source_power=normalize_to_source_power
		self.monitor_names=['fom_mon_{}'.format(i) for i in range(len(self.positions))]

		self.multi_freq_src=False

	def put_monitors(self,simulation):
		script=''
		for monitor_name,position in zip(self.monitor_names,self.positions):
			script+=ls.add_point_monitor_script(monitor_name,position)
		sim.fdtd.eval(script)

	def get_fom(self,simulation):
		fields=[ls.get_fields(simulation.solver_handle, monitor_name) for monitor_name in self.monitor_names]
		if self.normalize_to_source_power:
			source_power = np.zeros(np.shape(fields[0].wl))
			for i, wl in enumerate(fields[0].wl):
				source_power[i] = ls.get_source_power(simulation.solver_handle, wl=wl)
				self.source_power = source_power
				self.fields=fields
				pointfields=[field.getfield(field.x[0],field.y[0],field.z[0],self.wavelengths[0]) for field in fields]
		if self.weight_amplitudes is None:
			fom=sum([sum(pointfield*np.conj(pointfield)) for pointfield in pointfields])
		else:
			sum_of_pointfields=sum([pointfield*phase_factor for pointfield,phase_factor in zip(pointfields, self.weight_amplitudes)])
			fom=sum(sum_of_pointfields*np.conj(sum_of_pointfields))
		if self.normalize_to_source_power:
			fom=fom/np.array(source_power)
		return fom

	def add_adjoint_sources(self, sim):

		fields=self.fields
		pointfields=[field.getfield(field.x[0],field.y[0],field.z[0],self.wavelengths[0]) for field in fields]
		prefactor=1#eps0#*omega

		#print prefactor
		if self.weight_amplitudes is None:
			adjoint_sources=[prefactor*np.conj(pointfield) for pointfield in pointfields]
		else:
			pointfields = [field.getfield(field.x[0], field.y[0], field.z[0], self.wavelengths[0]) for field in fields]
			sum_of_pointfields=sum([pointfield*phase_factor for pointfield,phase_factor in zip(pointfields, self.weight_amplitudes)])
			prefactor=np.conj(sum_of_pointfields)
			adjoint_sources=[prefactor*phase_factor for phase_factor in self.weight_amplitudes]
		if self.normalize_to_source_power:
			adjoint_sources=adjoint_sources/self.source_power
			script=''
		for i,(adjoint_source,field) in enumerate(zip(adjoint_sources,fields)):
			script+=ls.add_dipole_script(field.x[0],field.y[0],field.z[0],self.wavelengths[0],adjoint_source,name_suffix=str(i))
			
		sim.fdtd.eval(script)
		return

	@staticmethod
	def add_dipole(sim,monitor_name,source_name):
		'''
		Add dipole at location of monitor
		'''
		pass




