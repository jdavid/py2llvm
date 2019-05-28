from py2llvm import _lib

Array = _lib.Array

def run(f, *args):
    args = f.call_args(*args)
    return _lib.run(f, args)
