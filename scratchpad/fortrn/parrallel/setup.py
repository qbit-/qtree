import os
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup

def configuration(parent_package='',top_path=None):
    config = Configuration(None,parent_package,top_path)
    libraries = [
            #'gomp',
            #'blas',
            ]
    config.add_extension('wrapper',
                         ['fortranmod.f95','wrapper.f95'],
                         libraries = libraries,
                         f2py_options = [],
                         define_macros = [('F2PY_REPORT_ON_ARRAY_COPY','1')],
                         # this is the flag gfortran needs to process OpenMP directives
                         extra_compile_args = ['-fopenmp'],
                         extra_link_args = [],
                         )
    return config

if __name__ == "__main__":
    setup(configuration=configuration)
