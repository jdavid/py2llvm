"""
Test the different ways to call "jit" and define signatures.
"""
import numpy as np
from pytest import mark

from py2llvm import Array, jit


#
# Type inference
#

def infer_pass(): pass
def infer_return(): return
def infer_none(): return None
def infer_int(): return 5
def infer_float(): return 5.0
def infer_name_int(): a = 5; return a
def infer_name_float(): a = 5.0; return a

@mark.parametrize("f", [infer_pass, infer_return, infer_none, infer_int,
                        infer_float, infer_name_int, infer_name_float])
def test_inference(f):
    fc = jit(f)
    assert f() == fc()


#
# Ways to define, or not, the signature
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

@jit
def f_dec(array):
    acc = 0.0
    i = 0
    while i < array.shape[0]:
        acc = acc + array[i]
        i = i + 1

    return acc

@jit
def f_dec_hints(array: Array(float, 1)) -> float:
    acc = 0.0
    i = 0
    while i < array.shape[0]:
        acc = acc + array[i]
        i = i + 1

    return acc

@jit(signature=signature)
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
    a = jit(f)             # guess signature from args
    assert not a.compiled
    a(array)
    assert a.compiled
    c = jit(f, signature)  # pass signature
    assert c.compiled
    b = jit(f_hints)       # with type hints
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
