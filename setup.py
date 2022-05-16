from setuptools import setup, find_packages

setup( name = 'mmdgpy',
       version = 1.0,
       author = 'Maximilian Hörl',
       author_email = 'maximilian.hoerl@mathematik.uni-stuttgart.de',
       url = 'https://gitlab.mathematik.uni-stuttgart.de/hoerlmn/mmdgpy',
       packages = find_packages(),
       install_requires = ['dune-mmesh', 'matplotlib', 'scipy'],
       python_requires = ">=3.7" )
