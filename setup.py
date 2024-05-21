# Setup script for Calipy
from setuptools import setup, find_packages

setup(
    name='calipy',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A description of your package.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='http://github.com/yourusername/calipy',
    packages=find_packages(),  # This will find all packages under calipy/
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
    install_requires=[
        # list your project dependencies here
        # e.g., 'numpy', 'pandas', etc.
    ],
    extras_require={
        'dev': [
            'pytest>=3.7',
        ],
    }
)

