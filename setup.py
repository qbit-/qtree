#!/usr/bin/env python

from setuptools import setup, find_packages
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='qtree',
    packages=find_packages(),
    description='Quantum circuits simulatior',
    long_description=long_description,
    long_description_content_type='text/markdown',

    install_requires=['cirq', 'networkx', 'mpi4py', 'tqdm'
                      , 'loguru'
                      , 'click'],
    entry_points={
        'console_scripts': [
            'qtree=qtree.cli:cli'
        ]
    },
    python_requires='>=3.3',
    include_package_data=True,
    license='GPLv2',
    classifiers=[],
)
