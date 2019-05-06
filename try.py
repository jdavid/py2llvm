import argparse

import numpy as np
import py2llvm as llvm
from py2llvm import float64, Array


def f(array):
    acc = 0.0
    i = 0
    while i < array.shape[0]:
        acc = acc + array[i]
        i = i + 1

    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    verbose = 2 if args.debug else 1

    array = [1.0, 2.5, 4.3]
    array = np.array(array, dtype=np.float64)
    print(array.dtype, array.shape)

    signature = Array(float64, 1), float64
    f = llvm.compile(f, signature, verbose=verbose)
    print('====== Output ======')
    f(array, debug=True)
