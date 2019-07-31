from llvmlite import binding, ir
import numpy as np

try:
    from py2llvm import _lib
except ImportError:
    _lib = None

from py2llvm import types


if _lib is None:
    class Function:
        def prepare(self, f):
            pass

        def __call__(self, *args, **kwargs):
            raise NotImplementedError

else:
    Array = _lib.Array
    Function = _lib.Function

    def expand_argument(py_arg, c_type):
        if isinstance(py_arg, Array):
            data = py_arg.addr
            size = py_arg.size
            return [data, size]

        return None


    def load_functions(module):
        sin_ft = ir.FunctionType(types.float64, (types.float64,))
        return {
            np.sin: ir.Function(module, sin_ft, name="fun"),
        }

    # Make functions available to LLVM
    binding.load_library_permanently(_lib.__file__)
