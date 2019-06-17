import argparse

import numpy as np
import py2llvm as llvm


@llvm.lazy
def f(array, out, n):
    for i in range(array.shape[0]):
        out[i] = np.sin(array[i]) * n


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args()
    verbose = args.verbose

    # Prepare function arguments
    array = [1.0, 2.5, 4.3]
    array = np.array(array, dtype=np.float64)
    out = np.empty((3,), dtype=np.float64)

    # Call (calls compile implicitly)
    print('>', array)

    f.py_function(array, out, 2)
    print('=', out)

    f(array, out, 2, verbose=verbose)
    print('=', out)
