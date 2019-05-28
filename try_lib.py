from py2llvm import Array
from py2llvm import lib
import py2llvm as llvm
import testa


def f(array: Array(float, 1)) -> float:
    acc = 0.0
    for i in range(array.shape[0]):
        acc = acc + array[i]
    return acc


if __name__ == '__main__':
    f = llvm.compile(f)

    array = lib.Array([1.0, 2.5, 4.3])
    x = testa.run(f, array)
    print(f'= {x}')
