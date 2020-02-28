import subprocess

from setuptools import setup


name = 'py2llvm'

def cmd(*args):
    cp = subprocess.run(args, stdout=subprocess.PIPE)
    if cp.returncode != 0:
        return None

    return cp.stdout.strip().decode().split()


setup(
    name=name,
    packages=[name],
    # Requirements
    install_requires=['icc-rt', 'llvmlite', 'numpy'],
    extras_require={
        'test': ['pytest', 'hypothesis'],
    },
)
