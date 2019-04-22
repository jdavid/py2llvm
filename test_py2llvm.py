from hypothesis import given
from hypothesis.strategies import integers

from py2llvm import llvm
import source


# Some constants
int64_min = -2**63
int64_max = 2**63-1
int32_min = -2**31
int32_max = 2**31-1


def test_no_args():
    fnames = 'ret_const', 'ret_var', 'ops'
    for fname in fnames:
        f = getattr(source, fname)
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


def test_loop():
    f = source.sum
    fc = llvm.compile(f)
    assert f() == fc()
