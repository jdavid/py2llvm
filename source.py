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
