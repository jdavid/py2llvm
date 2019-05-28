from setuptools import setup, Extension


name = 'py2llvm'

setup(
    name=name,
    packages=[name],
    # Requirements
    install_requires=['llvmlite', 'numpy'],
    extras_require={
        'test': ['pytest', 'hypothesis'],
    },
    # Extensions
    ext_modules=[
        Extension('py2llvm._lib', sources = ['py2llvm/_lib.c']),
    ],
)
