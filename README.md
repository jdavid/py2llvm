.. image:: https://travis-ci.org/jdavid/py2llvm.svg?branch=master
   :target: http://travis-ci.org/jdavid/py2llvm

This program translates a small subset of Python to LLVM IR.

Example:

    from typing import List
    import py2llvm as llvm

    def f(array, n):
        acc = 0.0
        i = 0
        while i < n:
            acc = acc + array[i]
            i = i + 1

        return acc

    signature = List[llvm.float64], llvm.int32, llvm.float64
    f = llvm.compile(f, signature)
    f(3, [1.0, 2.5, 4.3])

Notes:

- Only a very small subset is implemented.
- Only integers and floats are supported.
- Integers are translated to 64 bit integers in LLVM, and floats are translated
  to 64 bit doubles in LLVM.
- Function arguments and return values *must* be typed (type hints).

Types. The supported types are float32 float64, int32 and int64. It's also
possible to use Python's float and int types, they are considered aliases to
double and int64 respectively.

The compiled functions will be called using ctypes, so the allowed argument
types are a subset of the types available in ctypes.

Literals:

- The "2" literal is a 64 bits integer, the "2.0" literal is a double.
- No other literal is allowed (None, etc.)

Type conversion:

- Type conversion is done automatically, integers are converted to floats if
  one operand is a float.
- Type conversion can be explicit in an assignment, using type hints, e.g.
  "a: int = 2.0" the float literal will be converted to an integer.

Local variables:

- It's not possible to reuse the name of the same local variable for 2
  different types. For instance "a = 2; a = 2.0" is forbidden.

The return value:

- The functions *must* always return a value.
- The return value is converted as well, e.g. if the function is declared to
  return a float then in "return 2" the literal integer value will be converted
  to float.

LLVM is Static-Single-Assignment (SSA). There're 2 approaches to handle local
variables:

1. Follow the SSA form and use the PHI operator
2. Allocate local variables in the stack, to walk around the SSA form

We use the second solution (because it's simpler and the generated code looks
closer to the input Python source, so it's simpler to visually verify its
correctness). Then we run the "memory to register promotion" of the optimizer
(TODO).

The algorithm makes 2 passes to the AST. In the first pass the IR code blocks
are created, in the second pass the code is generated.

Useful links:

- https://docs.python.org/3.6/library/ast.html#abstract-grammar
- http://llvmlite.pydata.org/

About LLVM:

- https://mapping-high-level-constructs-to-llvm-ir.readthedocs.io/
- https://www.llvm.org/docs/tutorial/index.html
- https://llvm.org/docs/LangRef.html
