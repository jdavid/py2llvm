import argparse

import py2llvm as llvm


def f(n: int) -> int:
    a = [4, 2, 5] # Next: make this one a parameter

    acc = 0
    i = 0
    while i < n:
        acc = acc + a[i]
        i = i + 1

    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    verbose = 2 if args.debug else 1

    f = llvm.compile(f, verbose=verbose)
    print('====== Output ======')
    f(3, debug=True)
