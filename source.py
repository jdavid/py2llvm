def ret_const() -> float:
    return 42

def ret_var() -> float:
    a = 42
    return a

def ops() -> float:
    a = 2
    b = 4
    c = (a + b) * 2 - 3
    c = c / 3
    return c
