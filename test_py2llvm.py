from hypothesis import given
from hypothesis.strategies import integers, floats
from hypothesis.extra.numpy import arrays

from py2llvm import llvm
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

def test_for():
    f = source.sum
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
