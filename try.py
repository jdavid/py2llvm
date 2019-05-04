import argparse
from typing import List

import py2llvm as llvm


def f(array, n):
    acc = 0.0
    i = 0
    while i < n:
        acc = acc + array[i]
        i = i + 1

    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    verbose = 2 if args.debug else 1

    signature = List[llvm.float64], llvm.int32, llvm.float64
    f = llvm.compile(f, signature, verbose=verbose)
    print('====== Output ======')
    array = [1.0, 2.5, 4.3]
    f(array, len(array), debug=True)
