'''Implementation of a wrapper to nfft, which is a non uniform fast fourier transform. Non uniform in the 
sense that the nodes in real space are not evenly spaced. Wrapper is implemented using Cython. Functions 
that are defined here are taken from the 3.0 manual online. Hopefully 3.2 has the same interface. Will 
need to check if any new routines have been added.'''

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
	'''Copying a numpy array into a fftw_complex array. Not returning arrray but modifying input array 
	because space is probably already allocated.'''

	memcpy(arr_out, arr_in.data, n_elem*SIZEOF_COMPLEX)
	
cdef class NFFT:
	cdef cnfft.nfft_plan _c_nfft_plan
	cdef int d
	cdef int init_type
	cdef int fhat_set, f_set
	#These are commented out because this will not compile as the variables will be expanded by the macros
	#in nfft3.h and fftw3.h
	#cdef unsigned PRE_PHI_HUT, FG_PSI, PRE_LIN_PSI, PRE_FG_PSI, PRE_PSI, PRE_FULL_PSI
	#cdef unsigned MALLOC_X, MALLOC_F_HAT, MALLOC_F, FFT_OUT_OF_PLACE, FFTW_INIT, PRE_ONE_PSI

	def __init__(self):
		'''Python initialization'''
	
	def __cinit__(self):
		'''C Initialization. Set everything to 0'''
		self.d = 0
		self.init_type = 0
		self.f_set = 0
		self.fhat_set = 0
	
	def __dealloc__(self):
		'''We need to finalize the plan so that allocated memory is freed when this object is destroyed'''

		if self.fft_type != 0:
			self.finalize()

	@classmethod
	def init_1d(cls, int N1, int M):
	
		cdef NFFT self

		self = cls()

		if np.mod(N1,2) == 1:
			raise RuntimeError('N1 must be even')

		cnfft.nfft_init_1d(&(self._c_nfft_plan),N1,M)
		
		self.d = 1
		self.init_type = 1

		return self

	@classmethod
	def init_2d(cls, int N1, int N2, int M):

		cdef NFFT self

		self = cls()

		if np.mod(N1,2) == 1:
			raise RuntimeError('N1 must be even')

		if np.mod(N2,2) == 1:
			raise RuntimeError('N2 must be even')
		
		cnfft.nfft_init_2d(&(self._c_nfft_plan),N1,N2,M)
	
		self.d = 2
		self.init_type = 2

		return self

	@classmethod
	def init_3d(cls,int N1, int N2, int N3, int M):
		
		cdef NFFT self

		self = cls()

		if np.mod(N1,2) == 1:
			raise RuntimeError('N1 must be even')

		if np.mod(N2,2) == 1:
			raise RuntimeError('N2 must be even')
		
		if np.mod(N3,2) == 1:
			raise RuntimeError('N3 must be even')
		
		cnfft.nfft_init_3d(&(self._c_nfft_plan),N1,N2,N3,M)
		
		self.d = 3
		self.init_type = 3

		return self

	@classmethod
	def init(cls, int d, np.ndarray[DTYPEi_t,ndim=1] N, int M):

		cdef NFFT self

		self = cls()

		cnfft.nfft_init(&(self._c_nfft_plan), d, <int*>(N.data), M)
	
		self.d = d
		self.init_type = 4

		return self

	@classmethod
	def init_guru(cls,d, np.ndarray[DTYPEi_t,ndim=1] N, int M, np.ndarray[DTYPEi_t,ndim=1] n, int m, unsigned nfft_flags, unsigned fftw_flags):

		cdef NFFT self

		self = cls()

		cnfft.nfft_init_guru(&(self._c_nfft_plan), d, <int*>(N.data), M, <int*>(n.data), m, nfft_flags, fftw_flags)

		self.d = d
		self.init_type = 5

		return self

	def get_f(self):
		cdef np.ndarray[DTYPEc_t, ndim=1] f_out
		if self.f_set > 0:
			f_out = fftw_complex_to_numpy(self._c_nfft_plan.f,self.M_total)
			return f_out
		else:
			return 0

	def set_f(self,np.ndarray[DTYPEc_t, ndim=1] f_in):
		#Look at converting f_in to complex if it is not a complex type
		if self.init_type > 0:
			numpy_to_fftw_complex(f_in,self._c_nfft_plan.f,self.M_total)
			self.f_set = 1
		else:
			raise RuntimeError("Trying to set f before initializing NFFT plan.")
	
	def del_f(self):
		'''Do nothing right now'''	
	
	def get_fhat(self):
		cdef np.ndarray[DTYPEc_t, ndim=1] fhat_out
		if self.fhat_set > 0:
			fhat_out = fftw_complex_to_numpy(self._c_nfft_plan.f_hat,self.N_total)
			return fhat_out
		else:
			return 0

	def set_fhat(self,np.ndarray[DTYPEc_t, ndim=1] fhat_in):
		#Look at converting fhat_in to complex if it is not a complex type
		if self.init_type > 0:
			numpy_to_fftw_complex(fhat_in,self._c_nfft_plan.f_hat,self.N_total)
			self.fhat_set = 1
		else:
			raise RuntimeError("Trying to set f_hat before initializing NFFT plan.")
	
	def del_fhat(self):
		'''Do nothing right now'''

	f = property(get_f,set_f,del_f)
	f_hat = property(get_fhat,set_fhat,del_fhat)
	
	def set_nodes(self,x,y):
		'''Set the values for the real space nodes. x/y are shifted and scaled to be within the 2d torus: [-0.5,0.5). Not 
		sure if that is the correct thing to do'''
		
		cdef np.ndarray[DTYPE_t, ndim=1] xp = np.zeros(self.M_total, dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] yp = np.zeros(self.M_total, dtype=DTYPE)
		cdef double xmax, xmin, ymax, ymin
		cdef double xcent, ycent, xlength, ylength

		xmax = x.max()
		xmin = x.min()
		ymax = y.max()
		ymin = y.min()

		#nodes must be on the T^2 (2d torus). Values must be between [-0.5,0.5)^d
		#I center and scale x/y values to do this
		xcent = 0.5*(xmax + xmin)
		ycent = 0.5*(ymax + ymin)
		xlength = 1.001*(xmax - xmin) #1.001 so xmax is not mapped to 0.5
		ylength = 1.001*(ymax - ymin) #1.001 so ymax is not mapped to 0.5

		xp = (x-xcent) / xlength
		yp = (y-ycent) / ylength

		for j in range(self.M_total):
			self._c_nfft_plan.x[2*j] = xp[j]
			self._c_nfft_plan.x[2*j+1] = yp[j]

		#put in some flags here
		cnfft.nfft_precompute_one_psi(&(self._c_nfft_plan))

	def precompute_one_psi(self):
		cnfft.nfft_precompute_one_psi(&(self._c_nfft_plan))

	def adjoint(self):
		'''Do Adjoint transform. Calling generic adjoint routine. To run, must set f, by self.f = f, then call this routine. Output can be 
		accessed from self.f_hat'''
		self._check_adjoint()
		cnfft.nfft_adjoint(&(self._c_nfft_plan))
		self.fhat_set = 1

	def adjoint_1d(self):
		'''Do Adjoint transform. Calling adjoint_1d routine.'''
		self._check_adjoint()
		cnfft.nfft_adjoint_1d(&(self._c_nfft_plan))
		self.fhat_set = 1
	
	def adjoint_2d(self):
		'''Do Adjoint transform. Calling adjoint_2d routine.'''
		self._check_adjoint()
		cnfft.nfft_adjoint_1d(&(self._c_nfft_plan))
		self.fhat_set = 1
	
	def adjoint_3d(self):
		'''Do Adjoint transform. Calling adjoint_3d routine.'''
		self._check_adjoint()
		cnfft.nfft_adjoint_1d(&(self._c_nfft_plan))
		self.fhat_set = 1
	
	def trafo(self):
		'''Do the nfft transform from f_hat to f.'''
		self._check_trafo()
		cnfft.nfft_trafo(&(self._c_nfft_plan))
		self.f_set = 1
	
	def trafo_1d(self):
		'''Do the nfft transform from f_hat to f.'''
		self._check_trafo()
		cnfft.nfft_trafo_1d(&(self._c_nfft_plan))
		self.f_set = 1
	
	def trafo_2d(self):
		'''Do the nfft transform from f_hat to f.'''
		self._check_trafo()
		cnfft.nfft_trafo_2d(&(self._c_nfft_plan))
		self.f_set = 1
	
	def trafo_3d(self):
		'''Do the nfft transform from f_hat to f.'''
		self._check_trafo()
		cnfft.nfft_trafo_3d(&(self._c_nfft_plan))
		self.f_set = 1
	
	def check(self):
		cnfft.nfft_check(&(self._c_nfft_plan))

	def _check_adjoint(self):
		if self.f_set == 0:
			raise RuntimeError('Attempted to call an adjoint routine without setting f')

	def _check_trafo(self):
		if self.fhat_set == 0:
			raise RuntimeError('Attempted to cal a trafo routine without setting f_hat')

	def finalize(self):
		'''We need to finalize the plan so that allocated memory is freed when this object is destroyed'''

		cnfft.nfft_finalize(&(self._c_nfft_plan))
		self.init_type == 0
		self.d = 0
		self.f_set = 0
		self.fhat_set = 0

cdef class NNFFT:
	cdef cnfft.nnfft_plan _c_nnfft_plan
	cdef int d

	def __cinit__(self):

		self.d = 0
		
	@classmethod
	def init(cls, int d, np.ndarray[DTYPEi_t,ndim=1] N, int M):

		cdef NNFFT self

		self = cls()

		self.d = d

		cnfft.nnfft_init(&(self._c_nnfft_plan),d,self.N_total,M,<int*>(N.data))

	def __dealloc__(self):
		'''We need to finalize the plan so that allocated memory is freed when this object is destroyed'''
		cnfft.nnfft_finalize(&(self._c_nnfft_plan))

	
	def adjoint(self,f):

		cdef np.ndarray[DTYPEc_t, ndim=1] f_hat
		numpy_to_fftw_complex(f,self._c_nnfft_plan.f,self.M_total)
		cnfft.nnfft_adjoint(&(self._c_nnfft_plan))
		f_hat = fftw_complex_to_numpy(self._c_nnfft_plan.f_hat,self.N_total)
		return f_hat

	def trafo(self,f_hat):

		cdef np.ndarray[DTYPEc_t, ndim=1] f
		numpy_to_fftw_complex(f_hat,self._c_nnfft_plan.f_hat,self.N_total)
		cnfft.nnfft_trafo(&(self._c_nnfft_plan))
		f = fftw_complex_to_numpy(self._c_nnfft_plan.f,self.M_total)
		return f
