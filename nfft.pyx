'''Implementation of a wrapper to nfft, which is a non uniform fast fourier transform. Non uniform in the 
sense that the nodes in real space are not evenly spaced. Wrapper is implemented using Cython. Functions 
that are defined here are taken from the 3.0 manual online. Hopefully 3.2 has the same interface. Will 
need to check if any new routines have been added.'''

#NOTE
#arr.data may not work in the future with numpy array

cimport cnfft
from cnfft cimport fftw_complex

from libc.string cimport memcpy

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

DTYPEc = np.complex
ctypedef np.complex_t DTYPEc_t

DTYPEi = np.int
ctypedef np.int_t DTYPEi_t

cdef enum:
	SIZEOF_INT = sizeof(int)
	SIZEOF_FLOAT = sizeof(double)
	SIZEOF_COMPLEX = sizeof(fftw_complex)

#Flags that are used in the different initialization routines. They will be set to the values in nfft3.h
PRE_PHI_HUT = cnfft.PRE_PHI_HUT
FG_PSI = cnfft.FG_PSI
PRE_LIN_PSI = cnfft.PRE_LIN_PSI
PRE_FG_PSI = cnfft.PRE_FG_PSI
PRE_PSI = cnfft.PRE_PSI
PRE_FULL_PSI = cnfft.PRE_FULL_PSI
MALLOC_X = cnfft.MALLOC_X
MALLOC_F_HAT = cnfft.MALLOC_F_HAT
MALLOC_F = cnfft.MALLOC_F
FFT_OUT_OF_PLACE = cnfft.FFT_OUT_OF_PLACE
FFTW_INIT = cnfft.FFTW_INIT
NFFT_SORT_NODES = cnfft.NFFT_SORT_NODES
NFFT_OMP_BLOCKWISE_ADJOINT = cnfft.NFFT_OMP_BLOCKWISE_ADJOINT
MALLOC_V = cnfft.MALLOC_V #in NNFFT
NSDFT = cnfft.NSDFT #in NSFFT, I think
PRE_ONE_PSI = cnfft.PRE_ONE_PSI

#NFSFT flags
NFSFT_NORMALIZED = cnfft.NFSFT_NORMALIZED
NFSFT_USE_NDFT = cnfft.NFSFT_USE_NDFT
NFSFT_USE_DPT = cnfft.NFSFT_USE_DPT
NFSFT_MALLOC_X = cnfft.NFSFT_MALLOC_X
NFSFT_MALLOC_F_HAT = cnfft.NFSFT_MALLOC_F_HAT
NFSFT_MALLOC_F = cnfft.NFSFT_MALLOC_F
NFSFT_PRESERVE_F_HAT = cnfft.NFSFT_PRESERVE_F_HAT
NFSFT_PRESERVE_X = cnfft.NFSFT_PRESERVE_X
NFSFT_PRESERVE_F = cnfft.NFSFT_PRESERVE_F
NFSFT_DESTROY_F_HAT = cnfft.NFSFT_DESTROY_F_HAT
NFSFT_DESTROY_X = cnfft.NFSFT_DESTROY_X
NFSFT_DESTROY_F = cnfft.NFSFT_DESTROY_F
NFSFT_NO_DIRECT_ALGORITHM = cnfft.NFSFT_NO_DIRECT_ALGORITHM
NFSFT_NO_FAST_ALGORITHM = cnfft.NFSFT_NO_FAST_ALGORITHM
NFSFT_ZERO_F_HAT = cnfft.NFSFT_ZERO_F_HAT

#FPT flags
FPT_NO_STABILIZATION = cnfft.FPT_NO_STABILIZATION
FPT_NO_FAST_ALGORITHM = cnfft.FPT_NO_FAST_ALGORITHM
FPT_NO_DIRECT_ALGORITHM = cnfft.FPT_NO_DIRECT_ALGORITHM
FPT_PERSISTENT_DATA = cnfft.FPT_PERSISTENT_DATA
FPT_FUNCTION_VALUES = cnfft.FPT_FUNCTION_VALUES
FPT_AL_SYMMETRY = cnfft.FPT_AL_SYMMETRY

#NFSOFT flags
NFSOFT_NORMALIZED = cnfft.NFSOFT_NORMALIZED
NFSOFT_USE_NDFT = cnfft.NFSOFT_USE_NDFT
NFSOFT_USE_DPT = cnfft.NFSOFT_USE_DPT
NFSOFT_MALLOC_X = cnfft.NFSOFT_MALLOC_X
NFSOFT_REPRESENT = cnfft.NFSOFT_REPRESENT
NFSOFT_MALLOC_F_HAT = cnfft.NFSOFT_MALLOC_F_HAT
NFSOFT_MALLOC_F = cnfft.NFSOFT_MALLOC_F
NFSOFT_PRESERVE_F_HAT = cnfft.NFSOFT_PRESERVE_F_HAT
NFSOFT_PRESERVE_X = cnfft.NFSOFT_PRESERVE_X
NFSOFT_PRESERVE_F = cnfft.NFSOFT_PRESERVE_F
NFSOFT_DESTROY_F_HAT = cnfft.NFSOFT_DESTROY_F_HAT
NFSOFT_DESTROY_X = cnfft.NFSOFT_DESTROY_X
NFSOFT_DESTROY_F = cnfft.NFSOFT_DESTROY_F
NFSOFT_NO_STABILIZATION = cnfft.NFSOFT_NO_STABILIZATION
NFSOFT_CHOOSE_DPT = cnfft.NFSOFT_CHOOSE_DPT
NFSOFT_SOFT = cnfft.NFSOFT_SOFT
NFSOFT_ZERO_F_HAT = cnfft.NFSOFT_ZERO_F_HAT

#Solver flags
LANDWEBER = cnfft.LANDWEBER
STEEPEST_DESCENT = cnfft.STEEPEST_DESCENT
CGNR = cnfft.CGNR
CGNE = cnfft.CGNE
NORMS_FOR_LANDWEBER = cnfft.NORMS_FOR_LANDWEBER
PRECOMPUTE_WEIGHT = cnfft.PRECOMPUTE_WEIGHT
PRECOMPUTE_DAMP = cnfft.PRECOMPUTE_DAMP

cdef np.ndarray[DTYPEc_t] fftw_complex_to_numpy(fftw_complex *arr_in, int n_elem):
	'''Copying a fftw_complex array to a numpy array'''

	cdef np.ndarray[DTYPEc_t, ndim=1] arr = np.empty(n_elem,dtype=DTYPEc)

	memcpy(arr.data, arr_in, n_elem*SIZEOF_COMPLEX)

	return arr

cdef numpy_to_fftw_complex(np.ndarray[DTYPEc_t, ndim=1] arr_in, fftw_complex *arr_out, int n_elem):
	'''Copying a numpy array into a fftw_complex array. Not returning array but modifying input array 
	because space is probably already allocated.'''

	memcpy(arr_out, arr_in.data, n_elem*SIZEOF_COMPLEX)

cdef np.ndarray[DTYPE_t] real_to_numpy(double *arr_in, int n_elem):
	'''Copying a real array to a numpy array'''

	cdef np.ndarray[DTYPE_t, ndim=1] arr = np.empty(n_elem,dtype=DTYPE)

	memcpy(arr.data, arr_in, n_elem*SIZEOF_FLOAT)

	return arr

cdef numpy_to_real(np.ndarray[DTYPE_t] arr_in, double *arr_out, int n_elem):
	'''Copying a numpy array into a double array. Not returning array, but modifying input array
	because space for the array is probably already allocated.'''

	memcpy(arr_out, arr_in.data, n_elem*SIZEOF_FLOAT)

cdef class NFFT:
	cdef cnfft.nfft_plan _c_nfft_plan
	cdef int _init_type
	cdef int _fhat_set, _f_set, _x_set

	def __init__(self):
		'''Python initialization'''

	def __cinit__(self):
		'''C Initialization. Set everything to 0'''
		self._init_type = 0
		self._f_set = 0
		self._fhat_set = 0
		self._x_set = 0
	
	def __dealloc__(self):
		'''Finalize the plan so that allocated memory is freed when this object is destroyed. Calls self.finalize() if needed.'''

		if self._init_type != 0:
			self.finalize()

	@classmethod
	def init_1d(cls, int N1, int M):
		'''Initialization of a transform plan. Wrapper for d=1

		Parameters
		----------
		N1: int
			number of Fourier coefficients

		M: int
			number of real space nodes

		Returns
		-------
		self: NFFT
			Intialized NFFT class for 1D transform
		'''
	
		cdef NFFT self

		self = cls()

		if np.mod(N1,2) == 1:
			raise ValueError('N1 must be even')

		cnfft.nfft_init_1d(&(self._c_nfft_plan),N1,M)
		
		self._init_type = 1

		return self

	@classmethod
	def init_2d(cls, int N1, int N2, int M):
		'''Initialization of a transform plan. Wrapper for d=2

		Parameters
		----------
		N1: int
			number of Fourier coefficients in the 1st dimension

		N2: int
			number of Fourier coefficients in the 2nd dimension

		M: int
			number of real space nodes

		Returns
		-------
		self: NFFT
			Intialized NFFT class for 1D transform
		'''

		cdef NFFT self

		self = cls()

		if np.mod(N1,2) == 1:
			raise ValueError('N1 must be even')

		if np.mod(N2,2) == 1:
			raise ValueError('N2 must be even')
		
		cnfft.nfft_init_2d(&(self._c_nfft_plan),N1,N2,M)
	
		self._init_type = 2

		return self

	@classmethod
	def init_3d(cls,int N1, int N2, int N3, int M):
		'''Initialization of a transform plan. Wrapper for d=3

		Parameters
		----------
		N1: int
			number of Fourier coefficients in the 1st dimension
		
		N2: int
			number of Fourier coefficients in the 2nd dimension
		
		N3: int
			number of Fourier coefficients in the 3rd dimension

		M: int
			number of real space nodes

		Returns
		-------
		self: NFFT
			Intialized NFFT class for 2D transform
		'''
		
		cdef NFFT self

		self = cls()

		if np.mod(N1,2) == 1:
			raise ValueError('N1 must be even')

		if np.mod(N2,2) == 1:
			raise ValueError('N2 must be even')
		
		if np.mod(N3,2) == 1:
			raise ValueError('N3 must be even')
		
		cnfft.nfft_init_3d(&(self._c_nfft_plan),N1,N2,N3,M)
		
		self._init_type = 3

		return self

	@classmethod
	def init(cls, int d, np.ndarray N, int M):
		'''Initialization of a transform plan.

		Parameters
		----------
		d: int
			Number of dimensions for transform

		N: np.ndarray
			d element array with the number of Fourier coefficients in each dimension

		M: int
			Number of real space nodes
		'''

		cdef NFFT self

		if np.size(N) != d:
			raise ValueError("N should have d elements")

		for N_tmp in N:
			if np.mod(N_tmp,2) == 1:
				raise ValueError("Every member of N must be even")
		
		cdef np.ndarray[DTYPEi_t,ndim=1] N2 = np.array(N,dtype=DTYPEi)

		self = cls()

		cnfft.nfft_init(&(self._c_nfft_plan), d, <int*>(N2.data), M)
	
		self._init_type = 4

		return self

	@classmethod
	def init_guru(cls,d, np.ndarray[DTYPEi_t,ndim=1] N, int M, np.ndarray[DTYPEi_t,ndim=1] n, int m, unsigned nfft_flags, unsigned fftw_flags):

		cdef NFFT self

		self = cls()

		cnfft.nfft_init_guru(&(self._c_nfft_plan), d, <int*>(N.data), M, <int*>(n.data), m, nfft_flags, fftw_flags)

		self._init_type = 5

		return self

	property f:
		def __get__(self):
			cdef np.ndarray[DTYPEc_t, ndim=1] f_out
			if self._f_set > 0:
				f_out = fftw_complex_to_numpy(self._c_nfft_plan.f,self.M_total)
				return f_out
			else:
				return 0

		def __set__(self,np.ndarray f_in):
			cdef np.ndarray[DTYPEc_t, ndim=1] f_in2
			if self._init_type > 0:
				f_in2 = np.array(f_in,dtype=DTYPEc)
				numpy_to_fftw_complex(f_in2,self._c_nfft_plan.f,self.M_total)
				self._f_set = 1
			else:
				raise RuntimeError("Trying to set f before initializing NFFT plan.")
	
	property f_hat:
		def __get__(self):
			cdef np.ndarray[DTYPEc_t, ndim=1] fhat_out
			if self._fhat_set > 0:
				fhat_out = fftw_complex_to_numpy(self._c_nfft_plan.f_hat,self.N_total)
				return fhat_out
			else:
				return 0

		def __set__(self,np.ndarray fhat_in):
			cdef np.ndarray[DTYPEc_t, ndim=1] fhat_in2
			if self._init_type > 0:
				fhat_in2 = np.array(fhat_in,dtype=DTYPEc)
				numpy_to_fftw_complex(fhat_in2,self._c_nfft_plan.f_hat,self.N_total)
				self._fhat_set = 1
			else:
				raise RuntimeError("Trying to set f_hat before initializing NFFT plan.")

	property x:
		def __get__(self):
			cdef np.ndarray x_out
			if self._x_set > 0:
				x_out = real_to_numpy(self._c_nfft_plan.x,self.d*self.M_total)
				x_out = x_out.reshape((self.M_total,self.d))
				return x_out
			else:
				return 0

		def __set__(self,np.ndarray x_in):
			'''Set the nodes'''

			#x should be a (npts,ndim) array since it should then be stored in memory in the same linearized order that NFFT wants.
			if self._init_type > 0:
				if np.max(x_in) > 0.5 or np.min(x_in) < -0.5:
					raise ValueError("x must be in the interval [-0.5,0.5]^2")
				numpy_to_real(x_in,self._c_nfft_plan.x,self.d*self.M_total) #have M_total samples with d values in each sample
				self._x_set = 1
			else:
				raise RuntimeError("Trying to set the real space nodes before initializing NFFT plan")
	
	property N_total:
		def __get__(self):
			if self._init_type > 0:
				return self._c_nfft_plan.N_total
			else:
				return 0

	property M_total:
		def __get__(self):
			if self._init_type > 0:
				return self._c_nfft_plan.M_total
			else:
				return 0

	property d:
		def __get__(self):
			if self._init_type > 0:
				return self._c_nfft_plan.d
			else:
				return 0

	def precompute_one_psi(self):
		cnfft.nfft_precompute_one_psi(&(self._c_nfft_plan))

	def precompute(self):
		'''Does precomputation if flags for precomputation are set'''
		if self._c_nfft_plan.nfft_flags & PRE_ONE_PSI:
			self.precompute_one_psi()

	def adjoint(self):
		'''Do Adjoint transform. Calling generic adjoint routine. To run, must set f, by self.f = f, then call this routine. Output can be 
		accessed from self.f_hat'''
		self._check_adjoint()
		cnfft.nfft_adjoint(&(self._c_nfft_plan))
		self._fhat_set = 1

	def adjoint_1d(self):
		'''Do Adjoint transform. Calling adjoint_1d routine.'''
		self._check_adjoint()
		cnfft.nfft_adjoint_1d(&(self._c_nfft_plan))
		self._fhat_set = 1
	
	def adjoint_2d(self):
		'''Do Adjoint transform. Calling adjoint_2d routine.'''
		self._check_adjoint()
		cnfft.nfft_adjoint_1d(&(self._c_nfft_plan))
		self._fhat_set = 1
	
	def adjoint_3d(self):
		'''Do Adjoint transform. Calling adjoint_3d routine.'''
		self._check_adjoint()
		cnfft.nfft_adjoint_1d(&(self._c_nfft_plan))
		self._fhat_set = 1
	
	def trafo(self):
		'''Do the nfft transform from f_hat to f.'''
		self._check_trafo()
		cnfft.nfft_trafo(&(self._c_nfft_plan))
		self._f_set = 1
	
	def trafo_1d(self):
		'''Do the nfft transform from f_hat to f.'''
		self._check_trafo()
		cnfft.nfft_trafo_1d(&(self._c_nfft_plan))
		self._f_set = 1
	
	def trafo_2d(self):
		'''Do the nfft transform from f_hat to f.'''
		self._check_trafo()
		cnfft.nfft_trafo_2d(&(self._c_nfft_plan))
		self._f_set = 1
	
	def trafo_3d(self):
		'''Do the nfft transform from f_hat to f.'''
		self._check_trafo()
		cnfft.nfft_trafo_3d(&(self._c_nfft_plan))
		self._f_set = 1
	
	def check(self):
		cnfft.nfft_check(&(self._c_nfft_plan))

	def _check_adjoint(self):
		if self._f_set == 0:
			raise RuntimeError('Attempted to call an adjoint routine without setting f')

	def _check_trafo(self):
		if self._fhat_set == 0:
			raise RuntimeError('Attempted to call a trafo routine without setting f_hat')

	def finalize(self):
		'''We need to finalize the plan so that allocated memory is freed when this object is destroyed'''

		cnfft.nfft_finalize(&(self._c_nfft_plan))
		self.init_type == 0
		self._f_set = 0
		self._fhat_set = 0
		self._x_set = 0

cdef class NNFFT:
	cdef cnfft.nnfft_plan _c_nnfft_plan
	cdef int d, init_type
	cdef int f_set, fhat_set, x_set, v_set

	def __init__(self):
		'''Python initialization or something'''

	def __cinit__(self):

		self.d = 0
	
	def __dealloc__(self):
		'''We need to finalize the plan so that allocated memory is freed when this object is destroyed'''

		self.finalize()
		
	@classmethod
	def init(cls, int d, int N_total, int M_total, np.ndarray[DTYPEi_t,ndim=1] N):

		cdef NNFFT self

		self = cls()

		self.d = d

		cnfft.nnfft_init(&(self._c_nnfft_plan),d,N_total,M_total,<int*>(N.data))

		return self

	@classmethod
	def init_guru(cls, int d, int N_total, int M_total, np.ndarray[DTYPEi_t,ndim=1] N, np.ndarray[DTYPEi_t,ndim=1] N1, int m, unsigned nnfft_flags):

		cdef NNFFT self

		self = cls()

		self.d = d
		cnfft.nnfft_init_guru(&(self._c_nnfft_plan),d,N_total,M_total,<int*>(N.data),<int*>(N1.data),m,nnfft_flags)

		return self

	property f:
		def __get__(self):
			cdef np.ndarray[DTYPEc_t, ndim=1] f_out
			if self.f_set > 0:
				f_out = fftw_complex_to_numpy(self._c_nnfft_plan.f,self.M_total)
				return f_out
			else:
				return 0

		def __set__(self,np.ndarray[DTYPEc_t, ndim=1] f_in):
			#Look at converting f_in to complex if it is not a complex type
			if self.init_type > 0:
				numpy_to_fftw_complex(f_in,self._c_nnfft_plan.f,self.M_total)
				self.f_set = 1
			else:
				raise RuntimeError("Trying to set f before initializing NFFT plan.")
	
	property f_hat:
		def __get__(self):
			cdef np.ndarray[DTYPEc_t, ndim=1] fhat_out
			if self.fhat_set > 0:
				fhat_out = fftw_complex_to_numpy(self._c_nnfft_plan.f_hat,self.N_total)
				return fhat_out
			else:
				return 0

		def __set__(self,np.ndarray[DTYPEc_t, ndim=1] fhat_in):
			#Look at converting fhat_in to complex if it is not a complex type
			if self.init_type > 0:
				numpy_to_fftw_complex(fhat_in,self._c_nnfft_plan.f_hat,self.N_total)
				self.fhat_set = 1
			else:
				raise RuntimeError("Trying to set f_hat before initializing NFFT plan.")

	property x:
		def __get__(self):
			cdef np.ndarray x_out
			if self.x_set > 0:
				x_out = real_to_numpy(self._c_nnfft_plan.x,self.d*self.M_total)
				x_out = x_out.reshape((self.M_total,self.d))
				return x_out
			else:
				return 0

		def __set__(self,np.ndarray x_in):
			'''Set the nodes that are used for the real space values'''

			#x should be a (npts,ndim) array since it should then be stored in memory in the same linearized order that 
			#NFFT wants.
			if self.init_type > 0:
				if np.max(x_in) > 0.5 or np.min(x_in) < -0.5:
					raise ValueError("x must be in the interval [-0.5,0.5]^d")
				numpy_to_real(x_in,self._c_nnfft_plan.x,self.d*self.M_total) #have M_total samples with d values in each sample
				self.x_set = 1
			else:
				raise RuntimeError("Trying to set the real space nodes before initializing NFFT plan")

	property v:
		def __get__(self):
			cdef np.ndarray v_out
			if self.v_set > 0:
				v_out = real_to_numpy(self._c_nnfft_plan.v,self.d*self.N_total)
				v_out = v_out.reshape((self.N_total,self.d))
			else:
				return 0

		def __set__(self,np.ndarray v_in):
			'''Set the nodes that are used for the Fourier space values'''

			if self.init_type > 0:
				numpy_to_real(v_in,self._c_nnfft_plan.v,self.d*self.M_total) #have M_total samples with d values in each sample
				self.v_set = 1
			else:
				raise RuntimeError("Trying to set the frequency space nodes before initializing NFFT plan")
	
	property N_total:
		def __get__(self):
			if self.init_type > 0:
				return self._c_nfft_plan.N_total
			else:
				return 0

	property M_total:
		def __get__(self):
			if self.init_type > 0:
				return self._c_nfft_plan.M_total
			else:
				return 0

	def adjoint(self):
		self._check_adjoint()
		cnfft.nnfft_adjoint(&(self._c_nnfft_plan))
		self.fhat_set = 1
	
	def adjoint_direct(self):
		self._check_adjoint()
		cnfft.nnfft_adjoint_direct(&(self._c_nnfft_plan))
		self.fhat_set = 1

	def trafo(self):
		self._check_trafo()
		cnfft.nnfft_trafo(&(self._c_nnfft_plan))
		self.f_set = 1
	
	def trafo_direct(self):
		self._check_trafo()
		cnfft.nnfft_trafo_direct(&(self._c_nnfft_plan))
		self.f_set = 1

	def precompute_lin_psi(self):
		cnfft.nnfft_precompute_lin_psi(&(self._c_nnfft_plan))

	def precompute_psi(self):
		cnfft.nnfft_precompute_psi(&(self._c_nnfft_plan))

	def precompute_full_psi(self):
		cnfft.nnfft_precompute_full_psi(&(self._c_nnfft_plan))

	def precompute_phi_hut(self):
		cnfft.nnfft_precompute_phi_hut(&(self._c_nnfft_plan))
	
	def _check_adjoint(self):
		if self.f_set == 0:
			raise RuntimeError('Attempted to call an adjoint routine without setting f')

	def _check_trafo(self):
		if self.fhat_set == 0:
			raise RuntimeError('Attempted to call a trafo routine without setting f_hat')

	def finalize(self):
		if self.init_type > 0:
			cnfft.nnfft_finalize(&(self._c_nnfft_plan))
