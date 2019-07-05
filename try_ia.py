from array import array
import argparse
import ctypes
from ctypes import c_int, c_uint8, c_int32, c_size_t

from llvmlite import ir

import py2llvm as llvm
from py2llvm import StructType, int8p, int32, int64


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='count', default=0)
args = parser.parse_args()
verbose = args.verbose


"""
typedef struct {
  int ninputs;  // number of data inputs
  uint8_t* inputs[BLOSC2_PREFILTER_INPUTS_MAX];  // the data inputs
  int32_t input_typesizes[BLOSC2_PREFILTER_INPUTS_MAX];  // the typesizes for data inputs
  void *user_data;  // user-provided info (optional)
  uint8_t *out;  // automatically filled
  size_t out_size;  // automatically filled
  int32_t out_typesize;  // automatically filled
} blosc2_prefilter_params;

/**
 * @brief The type of the prefilter function.
 *
 * If the function call is successful, the return value should be 0; else, a negative value.
 */
#typedef int (*blosc2_prefilter_fn)(blosc2_prefilter_params* params);
"""

BLOSC2_PREFILTER_INPUTS_MAX = 128
class blosc2_prefilter_params(ctypes.Structure):
    _fields_ = [
        ('ninputs', c_int),
        ('inputs', ctypes.POINTER(c_uint8) * BLOSC2_PREFILTER_INPUTS_MAX),
        ('input_typesizes', c_int32 * BLOSC2_PREFILTER_INPUTS_MAX),
        ('user_data', ctypes.c_void_p),
        ('out', ctypes.POINTER(c_uint8)),
        ('out_size', c_size_t),
        ('out_typesize', c_int32),
    ]


class params_type_out:

    def __init__(self, ptr):
        self.ptr = ptr

    def subscript(self, visitor, value, slice):
        pass

class params_type(StructType):
    _name_ = 'blosc2_prefilter_params'
    _fields_ = [
        ('ninputs', int32), # int32 may not be the same as int
        ('inputs', ir.ArrayType(int8p, BLOSC2_PREFILTER_INPUTS_MAX)),
        ('input_typesizes', ir.ArrayType(int32, BLOSC2_PREFILTER_INPUTS_MAX)),
        ('user_data', int8p), # LLVM does not have the concept of void*
        ('out', int8p), # LLVM doesn't make the difference between signed and unsigned
        ('out_size', int64), # int64 may not be the same as size_t
        ('out_typesize', int32), # int32_t out_typesize;  // automatically filled
    ]

    @property
    def out(self):
        out = super().out()
        def cb(visitor):
            ptr = out(visitor)
            return params_type_out(ptr)

        return cb


@llvm.jit(verbose=verbose)
def f(params: params_type) -> int:
    n = params.out_size / params.out_typesize
    for i in range(n):
        params.out[i] = params.inputs[0][i]

    return 0


if __name__ == '__main__':
    a = array('f', [1.0, 2.0, 3.0])
    n = len(a)
    inputs = [a]
    out = array('f', [0.0] * n)

    data = blosc2_prefilter_params(
        len(inputs),                          # ninputs
        [x.buffer_info()[0] for x in inputs], # inputs
        [x.itemsize for x in inputs],         # input_typesizes
        0,                                    # user_data
        out.buffer_info()[0],                 # out
        n * out.itemsize,                     # out_size
        out.itemsize,                         # out_typesize
    )

    out = f(data)
    print(out)
