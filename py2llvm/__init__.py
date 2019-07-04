from .py2llvm import llvm
from .types import float32, float64, int32, int64, void
from .types import Array, Struct


jit = llvm.jit

__all__ = [
    'jit',                                          # Functions
    'float32', 'float64', 'int32', 'int64', 'void', # Basic types
    'Array', 'Struct',                              # Arrays
]
