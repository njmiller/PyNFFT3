from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
	name = 'PyNFFT',
	version = '0.1',
	cmdclass = {'build_ext': build_ext},
	ext_modules = [
		Extension("pynfft", ["nfft.pyx"],
			libraries=["nfft3","fftw3"])
		],
	include_dirs = [numpy.get_include()]
)

