"""
Tests for loops and arrays.
"""
from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays
from pytest import mark

from py2llvm import Array, jit


#
# Test literal arrays
#

def for_():
    acc = 0
    for x in [1, 2, 3, 4, 5]:
        acc = acc + x
    return acc

@mark.parametrize("f", [for_])
def test_for(f):
    f = jit(f)
    assert f() == f.py_function()


#
# Test numpy arrays
#

def np_1dim(array: Array(float, 1)) -> float:
    acc = 0.0
    i = 0
    while i < array.shape[0]:
        acc = acc + array[i]
        i = i + 1

    return acc

def for_range_start_step(array: Array(float, 1)) -> float:
    acc = 0.0
    for i in range(2, array.shape[0], 3):
        acc = acc + array[i]
    return acc

@mark.parametrize("f", [np_1dim, for_range_start_step])
@given(
    arrays(float, (5,), elements=floats(allow_nan=False, allow_infinity=False, width=32)),
)
def test_np_loop(f, a):
    fc = jit(f)
    assert fc(a) == f(a)


#
# Multidimensional arrays
#

def np_2dim(array: Array(float, 2)) -> float:
    acc = 0.0
    i = 0
    while i < array.shape[0]:
        j = 0
        while j < array.shape[1]:
            acc = acc + array[i,j]
            j = j + 1
        i = i + 1

    return acc

@mark.parametrize("f", [np_2dim])
@given(
    arrays(float, (3,2), elements=floats(allow_nan=False, allow_infinity=False, width=32)),
)
def test_ndim(f, a):
    fc = jit(f)
    assert f(a) == fc(a)


#
# Test output arrays
#

def np_assign(array: Array(float, 2), out: Array(float, 1)) -> int:
    i = 0
    while i < array.shape[1]:
        out[i] = 0.0
        j = 0
        while j < array.shape[0]:
            out[i] = out[i] + array[j,i]
            j = j + 1
        i = i + 1

    return 0

def for_range(array: Array(float, 2), out: Array(float, 1)) -> int:
    for i in range(array.shape[1]):
        out[i] = 0.0
        for j in range(array.shape[0]):
            out[i] = out[i] + array[j,i]

    return 0


def for_jit(array, out):
    acc = 0.0
    for i in range(array.shape[1]):
        out[i] = 0.0
        for j in range(array.shape[0]):
            out[i] = out[i] + array[j,i]
            acc = acc + array[j,i]

    return acc


@mark.parametrize("f", [np_assign, for_range, for_jit])
@given(
    arrays(float, (3,2), elements=floats(allow_nan=False, allow_infinity=False, width=32)),
    arrays(float, (3,), elements=floats()),
)
def test_out(f, a, b):
    fc = jit(f)
    assert f(a, b) == fc(a, b)
