from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
	name = 'NFFT',
	version = '0.1',
	cmdclass = {'build_ext': build_ext},
	ext_modules = [
		Extension("nfft", ["nfft.pyx"],
			libraries=["nfft3","fftw3"])
		],
	include_dirs = [numpy.get_include()]
)

