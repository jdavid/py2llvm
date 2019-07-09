import argparse
import ctypes

#import numpy as np
import py2llvm as llvm
#from py2llvm import StructType
from py2llvm import Struct, int64


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='count', default=0)
args = parser.parse_args()
verbose = args.verbose


#class x_type(StructType):
#    _name_ = 'point'
#    _fields_ = [
#        ('x', int),
#        ('y', int),
#    ]

x_type = Struct('point', x=int64, y=int64)

@llvm.jit(verbose=verbose)
def f(x: x_type) -> int64:
    return x.x * x.y


#@llvm.jit(verbose=verbose)
#def f(array, out, n):
#    for i in range(array.shape[0]):
#        out[i] = np.sin(array[i]) * n


class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


if __name__ == '__main__':
    out = f(Point(3, -5))
    print(out)

#  # Prepare function arguments
#  array = [1.0, 2.5, 4.3]
#  array = np.array(array, dtype=np.float64)
#  out = np.empty((3,), dtype=np.float64)

#  # Call (calls compile implicitly)
#  print('>', array)

#  f.py_function(array, out, 2)
#  print('=', out)

#  f(array, out, 2, verbose=verbose)
#  print('=', out)
