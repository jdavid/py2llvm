from py2llvm import _lib

Array = _lib.Array

def run(f, *args):
    args = f.call_args(*args)
    return _lib.run(f, args)


def expand_argument(py_arg, c_type):
    if isinstance(py_arg, Array):
        size = py_arg.size
        c_type = c_type._type_ * size
        arg = c_type.from_buffer(py_arg.data)
        return [arg, size]

    return None
