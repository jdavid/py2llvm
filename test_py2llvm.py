import hypothesis

from py2llvm import py2llvm
from run import run
import source


llvm_ir, sigs = py2llvm(open(f'source.py').read())

def test_no_args():
    fnames = 'ret_const', 'ret_var', 'ops'
    for fname in fnames:
        expected = getattr(source, fname)()
        actual = run(llvm_ir, fname, sigs[fname])
        assert expected == actual


int64_min = -2**63
int64_max = 2**63-1
@hypothesis.given(hypothesis.strategies.integers(int64_min/2, int64_max/2))
def test_int(x):
    fnames = 'double',
    for fname in fnames:
        expected = getattr(source, fname)(x)
        actual = run(llvm_ir, fname, sigs[fname], x)
        assert expected == actual


@hypothesis.given(hypothesis.strategies.integers(int64_min/2, int64_max/2))
def test_if_else(x):
    fname = 'if_else'
    expected = source.if_else(x)
    actual = run(llvm_ir, fname, sigs[fname], x)
    assert expected == actual
