#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup

setup(name='LumOpt',
      version='0.0.1',
      description='Continuous Adjoint Optimization wrapper for electromagnetic solvers',
      author='Christopher Lalau-Keraly',
      author_email='chriskeraly@gmail.com',
      install_requires=['numpy', 'scipy','matplotlib','pathlib'],
      packages=['lumopt']
      )