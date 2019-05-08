import argparse

import numpy as np
import py2llvm as llvm
from py2llvm import float64, int32, Array


def f(array, out):
    i = 0
    while i < array.shape[1]:
        out[i] = 0.0
        j = 0
        while j < array.shape[0]:
            out[i] = out[i] + array[j,i]
            j = j + 1
        i = i + 1

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    verbose = 2 if args.debug else 1

    array = [
        [1.0, 2.5, 4.3],
        [0.5, 2.0, 4.2],
    ]
    array = np.array(array, dtype=np.float64)
    out = np.empty((3,), dtype=np.float64)

    signature = Array(float64, 2), Array(float64, 1), int32
    fc = llvm.compile(f, signature, verbose=verbose)
    print('====== Output ======')
    fc(array, out, debug=True)
    print(out)
