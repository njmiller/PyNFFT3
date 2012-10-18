cdef extern from "fftw3.h":
	ctypedef double fftw_complex[2]

cdef extern from "nfft3.h":
	
	ctypedef enum:
		#Flags used in computations
		PRE_PHI_HUT
		FG_PSI
		PRE_LIN_PSI
		PRE_FG_PSI
		PRE_PSI
		PRE_FULL_PSI
		MALLOC_X
		MALLOC_F_HAT
		MALLOC_F
		FFT_OUT_OF_PLACE
		FFTW_INIT
		NFFT_SORT_NODES
		NFFT_OMP_BLOCKWISE_ADJOINT
		MALLOC_V
		NSDFT
		PRE_ONE_PSI

	#NFSFT flags
	ctypedef enum:
		NFSFT_NORMALIZED
		NFSFT_USE_NDFT
		NFSFT_USE_DPT
		NFSFT_MALLOC_X
		NFSFT_MALLOC_F_HAT
		NFSFT_MALLOC_F
		NFSFT_PRESERVE_F_HAT
		NFSFT_PRESERVE_X
		NFSFT_PRESERVE_F
		NFSFT_DESTROY_F_HAT
		NFSFT_DESTROY_X
		NFSFT_DESTROY_F
		NFSFT_NO_DIRECT_ALGORITHM
		NFSFT_NO_FAST_ALGORITHM
		NFSFT_ZERO_F_HAT

	#FPT flags
	ctypedef enum:
		FPT_NO_STABILIZATION
		FPT_NO_FAST_ALGORITHM
		FPT_NO_DIRECT_ALGORITHM
		FPT_PERSISTENT_DATA
		FPT_FUNCTION_VALUES
		FPT_AL_SYMMETRY

	#NFSOFT flags
	ctypedef enum:
		NFSOFT_NORMALIZED
		NFSOFT_USE_NDFT
		NFSOFT_USE_DPT
		NFSOFT_MALLOC_X
		NFSOFT_REPRESENT
		NFSOFT_MALLOC_F_HAT
		NFSOFT_MALLOC_F
		NFSOFT_PRESERVE_F_HAT
		NFSOFT_PRESERVE_X
		NFSOFT_PRESERVE_F
		NFSOFT_DESTROY_F_HAT
		NFSOFT_DESTROY_X
		NFSOFT_DESTROY_F
		NFSOFT_NO_STABILIZATION
		NFSOFT_CHOOSE_DPT
		NFSOFT_SOFT
		NFSOFT_ZERO_F_HAT

	#Solver flags
	ctypedef enum:
		LANDWEBER
		STEEPEST_DESCENT
		CGNR
		CGNE
		NORMS_FOR_LANDWEBER
		PRECOMPUTE_WEIGHT
		PRECOMPUTE_DAMP

#FFTW_DESTROY_INPUT
#FFTW_ESTIMATE
	
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
		unsigned nfft_flags #Flags for precomputation, (de)allocation, and FFTW usage

	void nfft_precompute_one_psi(nfft_plan *ths)
	void nfft_precompute_full_psi(nfft_plan *ths)
	void nfft_precompute_psi(nfft_plan *ths)
	void nfft_precompute_lin_psi(nfft_plan *ths)

	void nfft_init(nfft_plan *ths, int d, int *N, int M)
	void nfft_init_1d(nfft_plan *ths, int N1, int M)
	void nfft_init_2d(nfft_plan *ths, int N1, int N2, int M)
	void nfft_init_3d(nfft_plan *ths, int N1, int N2, int N3, int M)
	void nfft_init_guru(nfft_plan *ths, int d, int *N, int M, int *n, int m, unsigned nfft_flags, unsigned fftw_flags)
	
	void nfft_adjoint(nfft_plan *ths)
	void nfft_adjoint_1d(nfft_plan *ths)
	void nfft_adjoint_2d(nfft_plan *ths)
	void nfft_adjoint_3d(nfft_plan *ths)
	void nfft_trafo(nfft_plan *ths)
	void nfft_trafo_1d(nfft_plan *ths)
	void nfft_trafo_2d(nfft_plan *ths)
	void nfft_trafo_3d(nfft_plan *ths)

	void nfft_adjoint_direct(nfft_plan *ths)
	void nfft_trafo_direct(nfft_plan *ths)

	void nfft_check(nfft_plan *ths)
	void nfft_finalize(nfft_plan *ths)

	#NFCT function
	ctypedef struct nfct_plan:
		double *x #nodes in time/spatial domain
		int M_total #total number of samples
		int N_total #total number of Fourier coefficients
		double *f #vector of samples, size is M_total float types
		double *f_hat #vector of Fourier coefficients, since is N_total float_types
		int d #dimension, rank
		int *N #cut-off frequencies (kernel)
		unsigned nfct_flags

	void nfct_init_1d(nfct_plan *ths_plan, int N0, int M_total)
	void nfct_init_2d(nfct_plan *ths_plan, int N0, int N1, int M_total)
	void nfct_init_3d(nfct_plan *ths_plan, int N0, int N1, int N2, int M_total)
	void nfct_init(nfct_plan *ths_plan, int d, int *N, int M_total)
	void nfct_init_guru(nfct_plan *ths_plan, int d, int *N, int M_total, int *n, int m, unsigned nfct_flags, unsigned fftw_flags)
	
	void nfct_precompute_psi(nfct_plan *ths_plan)
	void nfct_trafo(nfct_plan *ths_plan)
	void nfct_trafo_direct(nfct_plan *ths_plan)
	void nfct_adjoint(nfct_plan *ths_plan)
	void nfct_adjoint_direct(nfct_plan *ths_plan)
	void nfct_finalize(nfct_plan *ths_plan)
	double nfct_phi_hut(nfct_plan *ths_plan, int k, int d)
	double nfct_phi(nfct_plan *ths_plan, double x, int d)
	
	#NFST function
	ctypedef struct nfst_plan:
		double *x #nodes in time/spatial domain
		int M_total #total number of samples
		int N_total #total number of Fourier coefficients
		double *f #vector of samples, size is M_total float types
		double *f_hat #vector of Fourier coefficients, since is N_total float_types
		#double complex *f #vector of samples, size is M_total float types
		#double complex *f_hat #vector of Fourier coefficients, since is N_total float_types
		int d #dimension, rank
		int *N #cut-off frequencies (kernel)
		unsigned nfst_flags

	void nfst_init_1d(nfst_plan *ths_plan, int N0, int M_total)
	void nfst_init_2d(nfst_plan *ths_plan, int N0, int N1, int M_total)
	void nfst_init_3d(nfst_plan *ths_plan, int N0, int N1, int N2, int M_total)
	void nfst_init(nfst_plan *ths_plan, int d, int *N, int M_total)
	void nfst_init_guru(nfst_plan *ths_plan, int d, int *N, int M_total, int *n, int m, unsigned nfst_flags, unsigned fftw_flags)
	
	void nfst_precompute_psi(nfst_plan *ths_plan)
	void nfst_trafo(nfst_plan *ths_plan)
	void nfst_trafo_direct(nfst_plan *ths_plan)
	void nfst_adjoint(nfst_plan *ths_plan)
	void nfst_adjoint_direct(nfst_plan *ths_plan)
	void nfst_finalize(nfst_plan *ths_plan)
	void nfst_full_psi(nfst_plan *ths_plan, double eps)
	double nfst_phi_hut(nfst_plan *ths_plan, int k, int d)
	double nfst_phi(nfst_plan *ths_plan, double x, int d)
	int nfst_fftw_2N(int n)
	int nfst_fftw_2N_rev(int n)

	#NNFFT functions
	ctypedef struct nnfft_plan:
		double *x #nodes (in time/spatial domain)
		double *v #nodes (in fourier domain)
		int M_total #total number of samples
		int N_total #total number of Fourier coefficients
		fftw_complex *f #vector of samples, size if M_total float_types
		fftw_complex *f_hat #vector of Fourier coefficients, size is N_total float_types
		#double complex *f #vector of samples, size if M_total float_types
		#double complex *f_hat #vector of Fourier coefficients, size is N_total float_types
		int d #dimension, rank
		double *sigma #oversampling-factor
	
	void nnfft_init(nnfft_plan *ths_plan, int d, int N_total, int M_total, int *N)
	void nnfft_init_guru(nnfft_plan *ths_plan, int d, int N_total, int M_total, int *N, int *N1, int m, unsigned nnfft_flags)
	void nnfft_trafo_direct(nnfft_plan *ths_plan)
	void nnfft_adjoint_direct(nnfft_plan *ths_plan)
	void nnfft_trafo(nnfft_plan *ths_plan)
	void nnfft_adjoint(nnfft_plan *ths_plan)
	void nnfft_precompute_lin_psi(nnfft_plan *ths_plan)
	void nnfft_precompute_psi(nnfft_plan *ths_plan)
	void nnfft_precompute_full_psi(nnfft_plan *ths_plan)
	void nnfft_precompute_phi_hut(nnfft_plan *ths_plan)
	void nnfft_finalize(nnfft_plan *ths_plan)

	#NSFFT functions
	ctypedef struct nsfft_plan:
		int M_total #total number of samples
		int N_total #total number of Fourier coefficients
		fftw_complex *f #vector of samples, size is M_total float types
		fftw_complex *f_hat #vector of Fourier coefficients, since is N_total float_types
		int d #dimension, ranl

	void nsfft_trafo_direct(nsfft_plan *ths)
	void nsfft_adjoint_direct(nsfft_plan *ths)
	void nsfft_trafo(nsfft_plan *ths)
	void nsfft_adjoint(nsfft_plan *ths)
	void nsfft_cp(nsfft_plan *ths, nfft_plan *ths_nfft)
	void nsfft_init_random_nodes_coeffs(nsfft_plan *ths)
	void nsfft_init(nsfft_plan *ths, int d, int J, int M, int m, unsigned flags)
	void nsfft_finalize(nsfft_plan *ths)

	#MRI functions
	ctypedef struct mri_inh_2d1d_plan:
		int M_total #total number of samples
		int N_total #total number of Fourier coefficients
		fftw_complex *f #vector of samples, size is M_total float types
		fftw_complex *f_hat #vector of Fourier coefficients, since is N_total float_types
		nfft_plan plan
		int N3
		double sigma3
		double *t
		double *w


	ctypedef struct mri_inh_3d_plan:
		int M_total #total number of samples
		int N_total #total number of Fourier coefficients
		fftw_complex *f #vector of samples, size is M_total float types
		fftw_complex *f_hat #vector of Fourier coefficients, since is N_total float_types
		nfft_plan plan
		int N3
		double sigma3
		double *t

	void mri_inh_2d1d_trafo(mri_inh_2d1d_plan *ths)
	void mri_inh_2d1d_adjoint(mri_inh_2d1d_plan *ths)
	void mri_inh_2d1d_init_guru(mri_inh_2d1d_plan *ths, int *N, int M, int *n, int m, double sigma, unsigned nfft_flags, unsigned fftw_flags)
	void mri_inh_2d1d_finalize(mri_inh_2d1d_plan *ths)
	void mri_inh_3d_trafo(mri_inh_3d_plan *ths)
	void mri_inh_3d_adjoint(mri_inh_3d_plan *ths)
	void mri_inh_3d_init_guru(mri_inh_3d_plan *ths, int *N, int M, int *n, int m, double sigma, unsigned nfft_flags, unsigned fftw_flags)
	void mri_inh_3d_finalize(mri_inh_3d_plan *ths)

	#NFSFT functions
	ctypedef struct nfsft_plan:
		int M_total #total number of samples
		int N_total #total number of Fourier coefficients
		fftw_complex *f #vector of samples, size is M_total float types
		fftw_complex *f_hat #vector of Fourier coefficients, since is N_total float_types

	void nfsft_init(nfsft_plan *plan, int N, int M)
	void nfsft_init_advanced(nfsft_plan *plan, int N, int M, unsigned int nfsft_flags)
	void nfsft_init_guru(nfsft_plan *plan, int N, int M, unsigned int nfsft_flags, unsigned int nfft_flags, int nfft_cutoff)
	void nfsft_precompute(int N, double kappa, unsigned int nfsft_flags, unsigned int fpt_flags)
	void nfsft_forget()
	void nfsft_trafo_direct(nfsft_plan *plan)
	void nfsft_adjoint_direct(nfsft_plan *plan)
	void nfsft_trafo(nfsft_plan *plan)
	void nfsft_adjoint(nfsft_plan *plan)
	void nfsft_finalize(nfsft_plan *plan)
	void nfsft_precompute_x(nfsft_plan *plan)

	#FPT functions
	ctypedef struct fpt_set:
		int flags
		int M
		int N
		int t

	fpt_set fpt_init(int M, int t, unsigned int flags)
	void fpt_precompute(fpt_set set, int m, double *alpha, double *beta, double *gam, int k_start, double threshold)
	void fpt_trafo_direct(fpt_set set, int m, fftw_complex *x, fftw_complex *y, int k_end, unsigned int flags)
	void fpt_trafo(fpt_set set, int m, fftw_complex *x, fftw_complex *y, int k_end, unsigned int flags)
	void fpt_transposed_direct(fpt_set set, int m, fftw_complex *x, fftw_complex *y, int k_end, unsigned int flags)
	void fpt_transposed(fpt_set set, int m, fftw_complex *x, fftw_complex *y, int k_end, unsigned int flags)
	void fpt_finalize(fpt_set set)

	#NFSOFT functions
	ctypedef struct nfsoft_plan:
		int M_total #total number of samples
		int N_total #total number of Fourier coefficients
		fftw_complex *f #vector of samples, size is M_total float types
		fftw_complex *f_hat #vector of Fourier coefficients, since is N_total float_types

	void nfsoft_precompute(nfsoft_plan *plan)
	fpt_set nfsoft_S03_single_fpt_init(int l, int k, int m, unsigned int flags, int kappa)
	void nfsoft_S03_fpt(fftw_complex *coeffs, fpt_set set, int l, int k, int m, unsigned int nfsoft_flags)
	void nfsoft_init(nfsoft_plan *plan, int N, int M)
	void nfsoft_init_advanced(nfsoft_plan *plan, int N, int M, unsigned int nfsoft_flags)
	void nfsoft_init_guru(nfsoft_plan *plan, int N, int M, unsigned int nfsoft_flags,unsigned int nfft_flags, int nfft_cutoff, int fpt_kapps)
	void nfsoft_trafo(nfsoft_plan *plan_nfsoft)
	void nfsoft_adjoint(nfsoft_plan *plan_nfsoft)
	void nfsoft_finalize(nfsoft_plan *plan_nfsoft)
	int nfsoft_posN(int n, int m, int B)

	#SOLVER functions
	ctypedef struct nfft_mv_plan_complex:
		int N_total
		int M_total
		fftw_complex *f_hat
		fftw_complex *f
	
	ctypedef struct solver_plan_complex:
		nfft_mv_plan_complex *mv
		double *w
		double *w_hat


	void solver_init_advanced_complex(solver_plan_complex *ths, nfft_mv_plan_complex *mv, unsigned flags)
	void solver_init_complex(solver_plan_complex *ths, nfft_mv_plan_complex *mv)
	void solver_before_loop_complex(solver_plan_complex *ths)
	void solver_loop_one_step_complex(solver_plan_complex *ths)
	void solver_finalize_complex(solver_plan_complex *ths)
	
	ctypedef struct nfft_mv_plan_double:
		int N_total
		int M_total
		double *f_hat
		double *f
	
	ctypedef struct solver_plan_double:
		nfft_mv_plan_double *mv
		double *w
		double *w_hat
	
	void solver_init_advanced_double(solver_plan_double *ths, nfft_mv_plan_double *mv, unsigned flags)

	#don't see these routines in the symbols in the library.
	#void solver_solver_init_double(solver_plan_double *ths, nfft_mv_plan_double *mv)
	#void solver_solver_before_loop_double(solver_plan_double *ths)
	#void solver_solver_loop_one_step_double(solver_plan_double *ths)
	#void solver_solver_finalize_double(solver_plan_double *ths)


