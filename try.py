import argparse

import numpy as np
import py2llvm as llvm
from py2llvm import float64, Array


@llvm.lazy
def f(array: Array(float64, 2), out: Array(float64, 1)) -> None:
    for i in range(array.shape[1]):
        out[i] = 0.0
        for j in range(array.shape[0]):
            out[i] = out[i] + array[j,i]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    verbose = 2 if args.debug else 1

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
