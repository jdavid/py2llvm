from .py2llvm import LLVM, Function, Signature, Parameter
from .py2llvm import llvm
from .types import float32, float64, void
from .types import int8, int8p, int16, int32, int64, int64p
from .types import Array, Struct, StructType


jit = llvm.jit

__all__ = [
    'jit',                             # Functions
    'float32', 'float64', 'void',      # Basic types
    'int8', 'int16', 'int32', 'int64', # Integers
    'int8p', 'int64p',                 # Pointers
    'Array',                           # Arrays
    'StructType', 'Struct',            # Structs
    # To override behaviour
    'LLVM', 'Function', 'Signature', 'Parameter',
]
