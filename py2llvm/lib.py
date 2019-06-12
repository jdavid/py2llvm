from llvmlite import binding, ir
import numpy as np
from py2llvm import _lib
from .py2llvm import float64


Array = _lib.Array
Function = _lib.Function


def run(f, *args):
    args = f.call_args(*args)
    f = Function(f)
    return f(args)


def expand_argument(py_arg, c_type):
    if isinstance(py_arg, Array):
        data = py_arg.addr
        size = py_arg.size
        return [data, size]

    return None


def load_functions(module):
    sin_ft = ir.FunctionType(float64, (float64,))
    return {
        np.sin: ir.Function(module, sin_ft, name="fun"),
    }

# Make functions available to LLVM
binding.load_library_permanently(_lib.__file__)
