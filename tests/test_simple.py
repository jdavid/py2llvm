"""
Test simple functions, not involvings loops.
"""

from hypothesis import given
from hypothesis.strategies import integers
from pytest import mark

from py2llvm import jit


#
# Different ways to return None (void)
#

def none1() -> None:
    pass

def none2() -> None:
    return

def none3() -> None:
    return None

@mark.parametrize("f", [none1, none2, none3])
def test_none(f):
    fc = jit(f)
    assert fc.py_function is f
    assert fc() is None


#
# Return scalars
#

def constant() -> float:
    return 42

def variable() -> float:
    a: float = 42
    return a

def operands() -> int:
    a: int = 2
    b: int = 4
    c: int = (a + b) * 2 - 3
    c: int = c / 3
    return c

@mark.parametrize("f", [constant, variable, operands])
def test_return(f):
    fc = jit(f)
    assert fc.py_function is f
    assert f() == fc()


#
# Functions with arguments
#

int64_min = -2**63
int64_max = 2**63-1
int32_min = -2**31
int32_max = 2**31-1


def double(x: int) -> int:
    return x * 2

def if_else(a: int) -> int:
    if a == 0:
        b = 5
    else:
        b = 2

    return a * b

@mark.parametrize("f", [double, if_else])
@given(integers(int64_min//2, int64_max//2))
def test_int(f, x):
    fc = jit(f)
    assert f(x) == fc(x)


@jit
def fibo(n: int) -> int:
    if n <= 1:
        return n

    return fibo(n-1) + fibo(n-2)

@given(integers(0, 20))
def test_call(x):
    """Test recursive calls."""
    assert fibo(x) == fibo.py_function(x)


@jit
def boolean(a: int, b: int, c: int) -> int:
    if (a < b and b < c) or c < a:
        return c * b

    if not (a < b):
        return b

    return a * 2

@given(
    integers(int32_min//2, int32_max//2),
    integers(int32_min//2, int32_max//2),
    integers(int32_min//2, int32_max//2),
)
def test_boolean(a, b, c):
    f = boolean
    assert f(a, b, c) == f.py_function(a, b, c)

