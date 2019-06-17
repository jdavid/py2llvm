from .py2llvm import Array, llvm
from .types import float32, float64, int32, int64, void


jit = llvm.jit

__all__ = [
    'jit',                                          # Functions
    'float32', 'float64', 'int32', 'int64', 'void', # Basic types
    'Array',                                        # Arrays
]
