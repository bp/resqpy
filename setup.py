""" Python Packaging information

This file allows the module to be pip-installed into a python kernel.

See https://packaging.python.org/tutorials/packaging-projects/

To install your working copy into your local conda environment in "editable mode"::

    pip install -e /path/to/working/copy

"""

from setuptools import setup

with open('README.md', 'r') as fh:
    long_descrption = fh.read()

setup(
    name='resqml',
    # version=1.0,
    description='Python API for working with RESQML files',
    long_descrption=long_descrption,
    long_description_content_type='text/markdown',
    # url='https://github.com/BP/resqpy',
    # author='',
    # author_email='',
    packages=['resqml'],
    include_package_data=True,
    classifiers=[
        # Need to put some info here for PyPi: language, licenses and OS. See https://pypi.org/classifiers/
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        # "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pandas',
        'h5py',
        'lasio',
    ],
)
