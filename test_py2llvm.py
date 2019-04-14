from hypothesis import given
from hypothesis.strategies import integers

from py2llvm import LLVM
import source


# Some constants
int64_min = -2**63
int64_max = 2**63-1
int32_min = -2**31
int32_max = 2**31-1


with open(f'source.py') as f:
    llvm = LLVM(f.read())


def test_no_args():
    fnames = 'ret_const', 'ret_var', 'ops'
    for fname in fnames:
        expected = getattr(source, fname)()
        actual = llvm.run(fname)
        assert expected == actual


@given(integers(int64_min//2, int64_max//2))
def test_int(x):
    fname = 'double'
    expected = getattr(source, fname)(x)
    actual = llvm.run(fname, x)
    assert expected == actual


@given(integers(int64_min/2, int64_max/2))
def test_if_else(x):
    fname = 'if_else'
    expected = getattr(source, fname)(x)
    actual = llvm.run(fname, x)
    assert expected == actual


@given(integers(0, 20))
def test_call(x):
    fname = 'fibo'
    expected = getattr(source, fname)(x)
    actual = llvm.run(fname, x)
    assert expected == actual


@given(
    integers(int32_min//2, int32_max//2),
    integers(int32_min//2, int32_max//2),
    integers(int32_min//2, int32_max//2),
)
def test_boolean(a, b, c):
    fname = 'boolean'
    expected = getattr(source, fname)(a, b, c)
    actual = llvm.run(fname, a, b, c)
    assert expected == actual


def test_loop():
    fname = 'sum'
    expected = getattr(source, fname)()
    actual = llvm.run(fname)
    assert expected == actual
