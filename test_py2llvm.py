from py2llvm import py2llvm
from run import run


def test_no_args():
    module = str(py2llvm(open(f'source.py').read()))

    import source
    fnames = 'ret_const', 'ret_var', 'ops'
    for fname in fnames:
        assert getattr(source, fname)() == run(module, fname)
