import subprocess

from setuptools import setup, Extension


name = 'py2llvm'

def ext_modules():
    cmd = ["pkg-config", "--variable=includedir", "libffi"]
    include_dir = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.strip()
    include_dirs = [include_dir.decode()]

    return [
        Extension('py2llvm._lib', sources=['py2llvm/_lib.c'],
                  include_dirs=include_dirs, libraries=['ffi'],
        ),
    ]

setup(
    name=name,
    packages=[name],
    # Requirements
    install_requires=['llvmlite', 'numpy'],
    extras_require={
        'test': ['pytest', 'hypothesis'],
    },
    # Entry points
    entry_points={
        'py2llvm_plugins': [
            'default = py2llvm.default',
            'lib = py2llvm.lib',
        ],
    },
    # Extensions
    ext_modules=ext_modules(),
)
