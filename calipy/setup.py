# Setup script for Calipy
from setuptools import setup, find_packages

setup(
    name = 'Calipy',
    version = '0.1.0',
    packages = find_packages(),
    install_requires = [
        # list of required packages
    ],
    author = 'Dr. Jemil Avers Butt',
    author_email = 'jemil.butt@atlasoptimization.com',
    description = 'A package for deep instrument models in engineering geodesy',
    keywords = 'engineering geodesy, probabilistic models, measurement instruments, calibration',
    url='http://github.com/atlasoptimization/calipy',
)