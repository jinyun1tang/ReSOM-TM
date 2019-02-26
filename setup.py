# from distutils.core import setup, Extension
#cython: language_level=3
from setuptools import setup, Extension

import numpy

import io

# If cython is available, the included cython *.pyx file
# is compiled, otherwise the *.c file is used
try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True


install_requires = [
    'numpy',
    'scipy'
]

extras_require = {
}

tests_require = ['nose']


cmdclass = { }
ext_modules = [ ]

if use_cython:
    ext_modules += [
        Extension("ReSOM.resom_mathlib",
                  [ "ReSOM/resom_mathlib.pyx"],
                  include_dirs=[numpy.get_include()]),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("ReSOM.resom_mathlib",
                  ["ReSOM/resom_mathlib.c"],
                  include_dirs=[numpy.get_include()]),
    ]

setup(
    name='ReSOM',
    version='1.0',
    packages=['ReSOM',],
    license='GNU General Public License v3.0',
    description='Reaction based model soil organic matter and microbes',
    author='Jinyun Tang',
    author_email='jinyuntang@gmail.com',
    url='https://github.com/jinyun1tang/ReSOM-TM.git/',
    long_description=io.open('README.md', 'r', encoding='utf-8').read(),
    keywords = 'soil organic matter',
    cmdclass = cmdclass,
    ext_modules=ext_modules,
    install_requires=install_requires,
    test_suite = 'nose.collector',
    tests_require = tests_require,
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: ',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 2.7',
    ],
)
