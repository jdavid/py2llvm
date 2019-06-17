from hypothesis import given
from hypothesis.strategies import integers, floats
from hypothesis.extra.numpy import arrays
import numpy as np

from py2llvm import llvm
from py2llvm import Array
from . import source


# Some constants
int64_min = -2**63
int64_max = 2**63-1
int32_min = -2**31
int32_max = 2**31-1


def test_constant():
    f = source.ret_const
    fc = llvm.jit(f)
    assert f() == fc()

def test_var():
    f = source.ret_var
    fc = llvm.jit(f)
    assert f() == fc()

def test_ops():
    f = source.ops
    fc = llvm.jit(f)
    assert f() == fc()

#
# Return None
#

def test_none():
    f = source.none1
    fc = llvm.jit(f)
    assert f() == fc()

    f = source.none2
    fc = llvm.jit(f)
    assert f() == fc()

    f = source.none3
    fc = llvm.jit(f)
    assert f() == fc()

#
# Arguments
#

@given(integers(int64_min//2, int64_max//2))
def test_int(x):
    f = source.double
    fc = llvm.jit(f)
    assert f(x) == fc(x)

@given(integers(int64_min/2, int64_max/2))
def test_if_else(x):
    f = source.if_else
    fc = llvm.jit(f)
    assert f(x) == fc(x)

@given(integers(0, 20))
def test_call(x):
    f = source.fibo
    fc = llvm.jit(f)
    assert f(x) == fc(x)

@given(
    integers(int32_min//2, int32_max//2),
    integers(int32_min//2, int32_max//2),
    integers(int32_min//2, int32_max//2),
)
def test_boolean(a, b, c):
    f = source.boolean
    fc = llvm.jit(f)
    assert f(a, b, c) == fc(a, b, c)


#
# Loops and arrays
#

def test_for():
    f = source.for_
    fc = llvm.jit(f)
    assert f() == fc()

@given(
    arrays(float, (3,), elements=floats(allow_nan=False, allow_infinity=False, width=32)),
)
def test_np_1dim(a):
    f = source.np_1dim
    fc = llvm.jit(f)
    assert f(a) == fc(a)

@given(
    arrays(float, (3,2), elements=floats(allow_nan=False, allow_infinity=False, width=32)),
)
def test_np_2dim(a):
    f = source.np_2dim
    fc = llvm.jit(f)
    assert f(a) == fc(a)

@given(
    arrays(float, (3,2), elements=floats(allow_nan=False, allow_infinity=False, width=32)),
    arrays(float, (3,), elements=floats()),
)
def test_np_assign(a, b):
    f = source.np_assign
    fc = llvm.jit(f)
    assert f(a, b) == fc(a, b)

@given(
    arrays(float, (3,2), elements=floats(allow_nan=False, allow_infinity=False, width=32)),
    arrays(float, (3,), elements=floats()),
)
def test_for_range(a, b):
    f = source.for_range
    fc = llvm.jit(f)
    assert f(a, b) == fc(a, b)

@given(
    arrays(float, (5,), elements=floats(allow_nan=False, allow_infinity=False, width=32)),
)
def test_for_range_start_step(a):
    f = source.for_range_start_step
    fc = llvm.jit(f)
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

@llvm.jit
def f_dec(array):
    acc = 0.0
    i = 0
    while i < array.shape[0]:
        acc = acc + array[i]
        i = i + 1

    return acc

@llvm.jit
def f_dec_hints(array: Array(float, 1)) -> float:
    acc = 0.0
    i = 0
    while i < array.shape[0]:
        acc = acc + array[i]
        i = i + 1

    return acc

@llvm.jit(signature)
def f_dec_sig(array):
    acc = 0.0
    i = 0
    while i < array.shape[0]:
        acc = acc + array[i]
        i = i + 1

    return acc


def test_calling_conventions():
    array = np.array([1.0, 2.5, 4.3])

    # Without decorators
    a = llvm.jit(f)             # guess signature from args
    assert not a.compiled
    a(array)
    assert a.compiled
    c = llvm.jit(f, signature)  # pass signature
    assert c.compiled
    b = llvm.jit(f_hints)       # with type hints
    assert b.compiled
    assert a(array) == b(array) == c(array) # compare

    # With decorators
    assert not f_dec.compiled   # guess signature from args
    f_dec(array)
    assert f_dec.compiled
    assert f_dec_sig.compiled   # with signature
    assert f_dec_hints.compiled # with type hints
    assert f_dec(array) == f_dec_sig(array) == f_dec_hints(array)

    assert a(array) == f_dec(array)


#
# Type inference
#

@given(
    arrays(float, (3,2), elements=floats(allow_nan=False, allow_infinity=False, width=32)),
)
def test_jit(a):
    f = source.jit
    fc = llvm.jit(f)

    b = np.zeros((3,), dtype=np.float64)
    c = np.zeros((3,), dtype=np.float64)
    assert f(a, b) == fc(a, c)
    assert np.array_equal(b, c)

def test_inference():
    names = [
        'infer_pass',
        'infer_return',
        'infer_none',
        'infer_int',
        'infer_float',
        'infer_name_int',
        'infer_name_float',
    ]

    for name in names:
        f = getattr(source, name)
        fc = llvm.jit(f)
        assert f() == fc()
