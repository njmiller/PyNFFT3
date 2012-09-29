cdef extern from "fftw3.h":
	ctypedef double fftw_complex[2]

cdef extern from "nfft3.h":
	#NFFT functions
	ctypedef struct nfft_plan:
		double *x #nodes in time/spatial domain
		int M_total #total number of samples
		int N_total #total number of Fourier coefficients
		fftw_complex *f #vector of samples, size is M_total float types
		fftw_complex *f_hat #vector of Fourier coefficients, since is N_total float_types
		#double complex *f #vector of samples, size is M_total float types
		#double complex *f_hat #vector of Fourier coefficients, since is N_total float_types
		int d #dimension, rank
		int *N #multi bandwidth

	void nfft_precompute_one_psi(nfft_plan *ths)

	void nfft_init(nfft_plan *ths, int d, int *N, int M)
	void nfft_init_1d(nfft_plan *ths, int N1, int M)
	void nfft_init_2d(nfft_plan *ths, int N1, int N2, int M)
	void nfft_init_3d(nfft_plan *ths, int N1, int N2, int N3, int M)

	void nfft_adjoint(nfft_plan *ths)
	void nfft_adjoint_1d(nfft_plan *ths)
	void nfft_adjoint_2d(nfft_plan *ths)
	void nfft_adjoint_3d(nfft_plan *ths)
	void nfft_trafo(nfft_plan *ths)
	void nfft_trafo_1d(nfft_plan *ths)
	void nfft_trafo_2d(nfft_plan *ths)
	void nfft_trafo_3d(nfft_plan *ths)

	void nfft_check(nfft_plan *ths)
	void nfft_finalize(nfft_plan *ths)

	#NNFFT functions
	ctypedef struct nnfft_plan:
		double *x #nodes (in time/spatial domain)
		double *v #nodes (in fourier domain)
		int M_total #total number of samples
		int N_total #total number of Fourier coefficients
		#fftw_complex *f #vector of samples, size if M_total float_types
		#fftw_complex *f_hat #vector of Fourier coefficients, size is N_total float_types
		double complex *f #vector of samples, size if M_total float_types
		double complex *f_hat #vector of Fourier coefficients, size is N_total float_types
		int d #dimension, rank
		double *sigma #oversampling-factor

	#NFSFT functions
