from distutils.core import setup
import os
from os.path import join

from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
from numpy.distutils.system_info import get_info
from numpy.distutils.misc_util import Configuration


def get_blas_info():
    def atlas_not_found(blas_info_):
        def_macros = blas_info.get('define_macros', [])
        for x in def_macros:
            if x[0] == "NO_ATLAS_INFO":
                # if x[1] != 1 we should have lapack
                # how do we do that now?
                return True
            if x[0] == "ATLAS_INFO":
                if "None" in x[1]:
                    # this one turned up on FreeBSD
                    return True
        return False

    blas_info = get_info('blas_opt', 0)
    if (not blas_info) or atlas_not_found(blas_info):
        cblas_libs = ['cblas']
        blas_info.pop('libraries', None)
    else:
        cblas_libs = blas_info.pop('libraries', [])
        return cblas_libs, blas_info


cblas_libs, blas_info = get_blas_info()

blas_include_dirs = blas_info['include_dirs']
del blas_info['include_dirs']

extensions = [
    Extension("ClusterCNN.custom_kmeans._k_means_elkan",
        sources=["_k_means_elkan.pyx"],
        include_dirs = [numpy.get_include()]
    ),
    Extension("ClusterCNN.custom_kmeans._k_means",
        libraries=cblas_libs,
        sources=["_k_means.pyx"],
        include_dirs = [
            '/home/andy/Documents/ClusterCNN/src/cblas',
            numpy.get_include(),
            *blas_include_dirs],
        extra_compile_args=blas_info.pop( 'extra_compile_args', []),
        **blas_info
    ),
]
setup(
    name = "Cython KMeans Build",
    ext_modules = cythonize(extensions),
)

