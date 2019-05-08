from py2llvm import Array


def ret_const() -> float:
    return 42


def ret_var() -> float:
    a: float = 42
    return a


def ops() -> int:
    a: int = 2
    b: int = 4
    c: int = (a + b) * 2 - 3
    c: int = c / 3
    return c


def double(x: int) -> int:
    return x * 2


def if_else(a: int) -> int:
    if a == 0:
        b = 5
    else:
        b = 2

    return a * b


def fibo(n: int) -> int:
    if n <= 1:
        return n

    return fibo(n-1) + fibo(n-2)


def boolean(a: int, b: int, c: int) -> int:
    if (a < b and b < c) or c < a:
        return c * b

    if not (a < b):
        return b

    return a * 2


def sum() -> int:
    acc = 0
    for x in [1, 2, 3, 4, 5]:
        acc = acc + x
    return acc


def np_1dim(array: Array(float, 1)) -> float:
    acc = 0.0
    i = 0
    while i < array.shape[0]:
        acc = acc + array[i]
        i = i + 1

    return acc


def np_2dim(array: Array(float, 2)) -> float:
    acc = 0.0
    i = 0
    while i < array.shape[0]:
        j = 0
        while j < array.shape[1]:
            acc = acc + array[i,j]
            j = j + 1
        i = i + 1

    return acc


def np_assign(array: Array(float, 2), out: Array(float, 1)) -> int:
    i = 0
    while i < array.shape[1]:
        out[i] = 0.0
        j = 0
        while j < array.shape[0]:
            out[i] = out[i] + array[j,i]
            j = j + 1
        i = i + 1

    return 0
