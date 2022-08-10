import os
from setuptools import setup, find_packages


# Helper function to load up readme as long description.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
        name = 'ssp_bayes_opt',
        version = '0.1',
        author='Michael Furlong', # TODO: look up multiple authors
        author_email='michael.furlong@uwaterloo.ca',
        description=('An implementation of an efficient Bayesian optimization algorithm using Spatial Semantic Pointers'),
        license = 'TBD',
        keywords = '',
        url='http://github.com/ctn-waterloo/ssp-bayesopt',
        packages=find_packages(),
        long_description=read('README.md'),
        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Programming Language :: Python :: 3',
            'Environment :: Console', 
            'Topic :: Scientific/Engineering :: Artificial Intelligence'
            ],
        install_requires=[
            'numpy>=1.21.2',
            'scipy',
            'sklearn',
            'pytest',
            'mypy',
            'typing-extensions'
            ]
)
