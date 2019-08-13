import subprocess

from setuptools import setup, Extension


name = 'py2llvm'

def cmd(*args):
    cp = subprocess.run(args, stdout=subprocess.PIPE)
    if cp.returncode != 0:
        return None

    return cp.stdout.strip().decode().split()


def lib_ext():
#   cmd = ["pkg-config", "--variable=includedir", "libffi"]
#   cp = subprocess.run(cmd, stdout=subprocess.PIPE)
#   if cp.returncode != 0:
#       return None

#   include_dirs = [cp.stdout.strip().decode()]
    return Extension(
        'py2llvm._lib',
        sources=['py2llvm/_lib.c'],
        #include_dirs=include_dirs,
        #libraries=['ffi'],
    )


def prefilter_ext():
    include_dirs = cmd("llvm-config", "--includedir")
    if include_dirs is None:
        return None

    return Extension(
        'py2llvm.prefilter',
        sources=['py2llvm/prefilter.c'],
        include_dirs=include_dirs + ['.'],
        libraries=['lib/prefilter'],
        library_dirs=['.'],
    )


def ext_modules():
    extensions = [lib_ext(), prefilter_ext()]
    return [x for x in extensions if x is not None]


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
