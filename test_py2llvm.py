from py2llvm import py2llvm
from run import run


def test_no_args():
    module = str(py2llvm(open(f'test_source.py').read()))

    from test_source import f, g
    assert f() == run(module, 'f')
    assert g() == run(module, 'g')
