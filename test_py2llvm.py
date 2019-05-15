from hypothesis import given
from hypothesis.strategies import integers, floats
from hypothesis.extra.numpy import arrays
import numpy as np

from py2llvm import llvm
from py2llvm import Array
import source


# Some constants
int64_min = -2**63
int64_max = 2**63-1
int32_min = -2**31
int32_max = 2**31-1


def test_constant():
    f = source.ret_const
    fc = llvm.compile(f)
    assert f() == fc()

def test_var():
    f = source.ret_var
    fc = llvm.compile(f)
    assert f() == fc()

def test_ops():
    f = source.ops
    fc = llvm.compile(f)
    assert f() == fc()

#
# Return None
#

def test_none():
    f = source.none1
    fc = llvm.compile(f)
    assert f() == fc()

    f = source.none2
    fc = llvm.compile(f)
    assert f() == fc()

    f = source.none3
    fc = llvm.compile(f)
    assert f() == fc()

#
# Arguments
#

@given(integers(int64_min//2, int64_max//2))
def test_int(x):
    f = source.double
    fc = llvm.compile(f)
    assert f(x) == fc(x)

@given(integers(int64_min/2, int64_max/2))
def test_if_else(x):
    f = source.if_else
    fc = llvm.compile(f)
    assert f(x) == fc(x)

@given(integers(0, 20))
def test_call(x):
    f = source.fibo
    fc = llvm.compile(f)
    assert f(x) == fc(x)

@given(
    integers(int32_min//2, int32_max//2),
    integers(int32_min//2, int32_max//2),
    integers(int32_min//2, int32_max//2),
)
def test_boolean(a, b, c):
    f = source.boolean
    fc = llvm.compile(f)
    assert f(a, b, c) == fc(a, b, c)


#
# Loops and arrays
#

def test_for():
    f = source.for_
    fc = llvm.compile(f)
    assert f() == fc()

@given(
    arrays(float, (3,), elements=floats(allow_nan=False, allow_infinity=False, width=32)),
)
def test_np_1dim(a):
    f = source.np_1dim
    fc = llvm.compile(f)
    assert f(a) == fc(a)

@given(
    arrays(float, (3,2), elements=floats(allow_nan=False, allow_infinity=False, width=32)),
)
def test_np_2dim(a):
    f = source.np_2dim
    fc = llvm.compile(f)
    assert f(a) == fc(a)

@given(
    arrays(float, (3,2), elements=floats(allow_nan=False, allow_infinity=False, width=32)),
    arrays(float, (3,), elements=floats()),
)
def test_np_assign(a, b):
    f = source.np_assign
    fc = llvm.compile(f)
    assert f(a, b) == fc(a, b)

@given(
    arrays(float, (3,2), elements=floats(allow_nan=False, allow_infinity=False, width=32)),
    arrays(float, (3,), elements=floats()),
)
def test_for_range(a, b):
    f = source.for_range
    fc = llvm.compile(f)
    assert f(a, b) == fc(a, b)

@given(
    arrays(float, (5,), elements=floats(allow_nan=False, allow_infinity=False, width=32)),
)
def test_for_range_start_step(a):
    f = source.for_range_start_step
    fc = llvm.compile(f)
    assert f(a) == fc(a)

#
# Calling conventions
#

signature = Array(float, 1), float

def f(array):
    acc = 0.0
    i = 0
    while i < array.shape[0]:
        acc = acc + array[i]
        i = i + 1

    return acc

def f_hints(array: Array(float, 1)) -> float:
    acc = 0.0
    i = 0
    while i < array.shape[0]:
        acc = acc + array[i]
        i = i + 1

    return acc

@llvm.compile(signature)
def f_dec(array):
    acc = 0.0
    i = 0
    while i < array.shape[0]:
        acc = acc + array[i]
        i = i + 1

    return acc

@llvm.lazy(signature)
def f_dec_lazy(array):
    acc = 0.0
    i = 0
    while i < array.shape[0]:
        acc = acc + array[i]
        i = i + 1

    return acc

@llvm.compile
def f_dec_hints(array: Array(float, 1)) -> float:
    acc = 0.0
    i = 0
    while i < array.shape[0]:
        acc = acc + array[i]
        i = i + 1

    return acc

@llvm.lazy
def f_dec_hints_lazy(array: Array(float, 1)) -> float:
    acc = 0.0
    i = 0
    while i < array.shape[0]:
        acc = acc + array[i]
        i = i + 1

    return acc


def test_calling_conventions():
    array = np.array([1.0, 2.5, 4.3])

    # With type hints
    a = llvm.compile(f_hints)
    b = llvm.lazy(f_hints)
    c = llvm.lazy(f_hints)
    c.compile()
    assert a(array) == b(array) == c(array)

    # With signature
    a = llvm.compile(f, signature)
    b = llvm.lazy(f, signature)
    c = llvm.lazy(f, signature)
    c.compile()
    assert a(array) == b(array) == c(array)

    # Decorators
    assert f_dec(array) == f_dec_lazy(array) == f_dec_hints(array) == f_dec_hints_lazy(array)


#
# Type inference
#

@given(
    arrays(float, (3,2), elements=floats(allow_nan=False, allow_infinity=False, width=32)),
    arrays(float, (3,), elements=floats()),
)
def test_jit(a, b):
    f = source.jit
    fc = llvm.lazy(f)
    assert f(a, b) == fc(a, b)
