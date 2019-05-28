from .py2llvm import llvm
from .py2llvm import float32, float64, int32, int64, Array


lazy = llvm.lazy
compile = llvm.compile

__all__ = [
    'compile', 'lazy',                      # Functions
    'float32', 'float64', 'int32', 'int64', # Scalar types
    'Array',                                # Arrays
]
