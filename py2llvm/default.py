def expand_argument(py_arg, c_type):
    # Scalar
    if isinstance(py_arg, (int, float)):
        return [py_arg]

    # Numpy
    array_interface = getattr(py_arg, '__array_interface__', None)
    if array_interface is not None:
        array_interface = py_arg.__array_interface__
        data = array_interface['data'][0]
        shape = array_interface['shape']
        return [data] + [n for n in shape]

    return None
