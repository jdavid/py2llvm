from py2llvm import _lib


Array = _lib.Array
Function = _lib.Function


def run(f, *args):
    args = f.call_args(*args)
    f = Function(f)
    return f(args)


def expand_argument(py_arg, c_type):
    if isinstance(py_arg, Array):
        data = py_arg.addr
        size = py_arg.size
        #c_type = c_type._type_ * size
        #data= c_type.from_buffer(py_arg.data)
        return [data, size]

    return None
