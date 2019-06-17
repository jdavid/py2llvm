from .py2llvm import Array, llvm
from .types import float32, float64, int32, int64, void


lazy = llvm.lazy
compile = llvm.compile

__all__ = [
    'compile', 'lazy',                              # Functions
    'float32', 'float64', 'int32', 'int64', 'void', # Basic types
    'Array',                                        # Arrays
]
