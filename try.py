import argparse

from py2llvm import llvm


def f() -> int:
    acc = 0
    for x in [1, 2, 3, 4, 5]:
        acc = acc + x
    return acc


source = """
def g() -> int:
    acc = 0
    for x in [1, 2, 3, 4, 5]:
        acc = acc + x
    return acc
"""



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    debug = args.debug

#   print('====== Source ======')
#   llvm(source, debug=debug)
#   g = llvm['g']
#   print(source)
#   print('====== IR ======')
#   print(llvm.ir)
#   print('====== Output ======')
#   g(debug=True)

    print('====== Source ======')
    f = llvm(f, debug=debug)
    print(f.py_source)
    print('====== IR ======')
    print(f.ir)
    print('====== Output ======')
    f(debug=True)
