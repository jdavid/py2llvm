from py2llvm import py2llvm
from run import run


def test_no_args():
    llvm_ir, sigs = py2llvm(open(f'source.py').read())

    import source
    fnames = 'ret_const', 'ret_var', 'ops'
    for fname in fnames:
        expected = getattr(source, fname)()
        assert expected == run(llvm_ir, fname, sigs[fname])
