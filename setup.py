from distutils.core import setup, Extension


setup(
    name='testa',
    ext_modules=[
        Extension('_testa', sources = ['_testa.c']),
    ],
)
