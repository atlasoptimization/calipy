# Setup script for Calipy
from setuptools import setup, find_packages

setup(
    name='calipy',
    version='0.5.0',
    author='Dr. Jemil Avers Butt',
    author_email='jemil.butt.@atlasoptimization.com',
    description='CaliPy, the calibration library python. A library for building and solving probabilistic instrument models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='http://github.com/atlasoptimization/calipy',
    packages=find_packages(),  # This will find all packages under calipy/
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Scientists, Developers',
        'License :: Prosperity Public License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
    install_requires=[
        'torch>=2.0',
        'pyro-ppl>=1.8.6',
        'pandas>=2.0.0',
        'matplotlib>=3.3.0',
        'einops>=0.8.1',
        'torchviz>=0.0.2',
        'varname>=0.8.1',
        'networkx>=2.4',
    ],
    extras_require={
        'dev': [
            'pytest>=3.7',
        ],
    }
)

