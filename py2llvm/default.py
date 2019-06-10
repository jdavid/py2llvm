try:
    import numpy as np
except ImportError:
    np = None


def expand_argument(py_arg, c_type):
    # Scalar
    if isinstance(py_arg, (int, float)):
        return [py_arg]

    # List
    if isinstance(py_arg, list):
        size = len(py_arg)
        c_type = c_type._type_ * size
        arg = c_type(*py_arg)
        return [arg, size]

    # Numpy
    if np is not None and isinstance(py_arg, np.ndarray):
        # NumPy array
        data = py_arg.__array_interface__['data'][0]
        size = py_arg.size
        return [data] + [n for n in py_arg.shape]

    return None
