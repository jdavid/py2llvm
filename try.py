import argparse

import numpy as np
import py2llvm as llvm


@llvm.lazy
def f(array, out):
    for i in range(array.shape[1]):
        out[i] = 0.0
        for j in range(array.shape[0]):
            out[i] = out[i] + array[j,i]

    a = 5
    return a


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args()
    verbose = args.verbose

    # Prepare function arguments
    array = [
        [1.0, 2.5, 4.3],
        [0.5, 2.0, 4.2],
    ]
    array = np.array(array, dtype=np.float64)
    out = np.empty((3,), dtype=np.float64)

    # Call (calls compile implicitly)
    f(array, out, verbose=verbose)
    print(out)
