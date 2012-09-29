cdef extern from "fftw3.h":
	ctypedef double fftw_complex[2]

cdef extern from "nfft3.h":
	ctypedef struct nfft_plan:
		double *x
		int M_total
		int N_total
		fftw_complex *f
		fftw_complex *f_hat

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
