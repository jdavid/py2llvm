import argparse
import inspect

import py2llvm as llvm

double = llvm.double
def f() -> double:
    acc = 0
    for x in [1, 2, 3, 4, 5]:
        acc = acc + x
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    debug = args.debug

    print('====== Source ======')
    print(inspect.getsource(f))
    f = llvm.compile(f, debug=debug)
    print('====== IR ======')
    print(f.ir)
    print('====== Output ======')
    f(3, debug=True)
