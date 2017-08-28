from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


ext_modules=[
    Extension('native_interp_to_u',
              sources=['native_interp_to_u.pyx'],
              include_dirs=[numpy.get_include()]),
]

setup(
    name = 'native_interp_to_u',
    ext_modules = cythonize(ext_modules)
)

