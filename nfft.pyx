'''Implementation of a wrapper to nfft, which is a non uniform fast fourier transform. Non uniform in the 
sense that the nodes in real space are not evenly spaced. Wrapper is implemented using Cython'''

cimport cnfft

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

DTYPEc = np.complex
ctypedef np.complex_t DTYPEc_t

DTYPEi = np.int
ctypedef np.int_t DTYPEi_t

cdef class NFFT_2D:
	cdef cnfft.nfft_plan _c_nfft_plan
	cdef int N1, N2, N_total, M_total
	def __cinit__(self,N1=100,N2=100,M=100):

		#N1 and N2 must be even for a NFFT. Possibly could change to just give an error
		if np.mod(N1,2) == 1:
			N1 = N1 + 1

		if np.mod(N2,2) == 1:
			N2 = N2 + 1

		cnfft.nfft_init_2d(&(self._c_nfft_plan),N1,N2,M)
		self.N1 = N1
		self.N2 = N2
		self.N_total = N1*N2
		self.M_total = M

	def __dealloc__(self):
		'''We need to finalize the plan so that allocated memory is freed when this object is destroyed'''

		cnfft.nfft_finalize(&(self._c_nfft_plan))

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

#	def show_nodes(self):
#
#		cdef int j
#
#		for j in range(2*self.M_total):
#			print self._c_nfft_plan.x[j]

	def adjoint(self,f,x=None,y=None):
		'''Do Adjoint transform'''

		cdef np.ndarray[DTYPEc_t, ndim=2] f_hat = np.zeros([self.N1,self.N2], dtype=DTYPEc)
		cdef int i, j, k

		#NFFT has some precomputation stuff it can do. User guide usually invokes it conditionally which could
		#be added later when I understand more about what it actually does. Right now I am just calling the routine
		cnfft.nfft_precompute_one_psi(&(self._c_nfft_plan))

		#adjoint transform takes the values in nfft_plan.f, finds the Fourier transform through its algorithm, and 
		#outputs the elements in nfft_plan.f_hat
		for i in range(self.M_total):
			self._c_nfft_plan.f[i][0] = f[i].real
			self._c_nfft_plan.f[i][1] = f[i].imag

		cnfft.nfft_adjoint_2d(&(self._c_nfft_plan))

		#nfft stores f_hat data in some linearized form k=\sum_{t=0}^{d-1} (k_t + N_t/2) \prod_{t'=t+1}^{d-1} N_{t'}
		#want to convert it back to 2d image
		for i in range(self.N1):
			for j in range(self.N2):
				k = i*self.N2 + j
				f_hat[i,j] = self._c_nfft_plan.f_hat[k][0] + self._c_nfft_plan.f_hat[k][1]*1.0j


		return f_hat

	def nfft_trafo(self,f_hat):
		'''Do the nfft transform from f_hat to f. NOT TESTED'''

		cdef np.ndarray[DTYPE_t, ndim=1] f = np.zeros(self.M_total, dtype=DTYPE)
		cdef int i, j, k

		cnfft.nfft_precompute_one_psi(&(self._c_nfft_plan))

		for i in range(self.N1):
			for j in range(self.N2)
				k = i*self.N2 + j
				self._c_nfft_plan.f_hat[k][0] = f_hat[i,j].real
				self._c_nfft_plan.f_hat[k][1] = f_hat[i,j].imag

		cnfft.nfft_trafo_2d(&(self._c_nfft_plan))

		f = self._c_nfft_plan.f

		return f


