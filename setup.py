from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
	name = 'NFFT',
	version = '0.1',
	cmdclass = {'build_ext': build_ext},
	ext_modules = [
		Extension("nfft", ["nfft.pyx"],
			libraries=["nfft3","fftw3"])
		]
)

