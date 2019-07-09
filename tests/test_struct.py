import ctypes

from hypothesis import given
from hypothesis.strategies import integers

import py2llvm as llvm
from py2llvm import Struct



x_type = Struct('point', x=int, y=int)

class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

@llvm.jit
def f(x: x_type) -> int:
    return x.x * x.y


int64_min = -2**63
int64_max = 2**63-1
int32_min = -2**31
int32_max = 2**31-1

@given(
    integers(int32_min, int32_max),
    integers(int32_min, int32_max),
)
def test_point(a, b):
    p = Point(a, b)
    assert f(p) == f.py_function(p)
