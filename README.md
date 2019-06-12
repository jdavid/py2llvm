[![Build Status](https://travis-ci.org/jdavid/py2llvm.svg?branch=master)](http://travis-ci.org/jdavid/py2llvm)

This program translates a small subset of Python to LLVM IR.

Example:

    import numpy as np
    import py2llvm as llvm

    @llvm.lazy
    def f(array):
        acc = 0.0
        for i in range(array.shape[0]):
            acc = acc + array[i]

        return acc

    array = [1.0, 2.5, 4.3]
    array = np.array(array, dtype=np.float64)
    print(f(array))

Only a very small subset of Python is supported: i.e. use Numba instead. Maybe
the main technical difference with Numba is that Numba translates to LLVM from
Python bytecode, while here I use the AST.

The compiled functions will be called using libffi.

Signature. The function signature can be explicitely given, or will be defined
with the first call to the function. If the return type is not specified it
will be inferred (following some very simple rules.

Types. The supported types are float32 float64, int32, int64 and numpy arrays
(or any array providing the \_\_array\_interface\_\_).  It's also possible to
use Python's float and int types, they are considered aliases to float64 and
int64 respectively.

Literals. The `2` literal is a 64 bits integer, the `2.0` literal is a double.
No other literal is allowed (None, etc.)

Type conversion:

- Type conversion is done automatically, integers are converted to floats if
  one operand is a float.
- Type conversion can be explicit in an assignment, using type hints, e.g.
  "a: int = 2.0" the float literal will be converted to an integer.

The return value:

- The return value is converted as well, e.g. if the function is declared to
  return a float then in "return 2" the literal integer value will be converted
  to float.

Local variables:

- It's not possible to reuse the name of the same local variable for 2
  different types. For instance "a = 2; a = 2.0" is forbidden.

LLVM is Static-Single-Assignment (SSA). There're 2 approaches to handle local
variables:

1. Follow the SSA form and use the PHI operator
2. Allocate local variables in the stack, to walk around the SSA form

We use the second solution (because it's simpler and the generated code looks
closer to the input Python source, so it's simpler to visually verify its
correctness). Then the code is optimized, so memory is promoted to registers.

The algorithm makes 3 passes to the AST: in the first pass the return type is
inferred; in the second pass the IR code blocks are created; in the third pass
the code is generated.

Useful links:

- https://docs.python.org/3.6/library/ast.html#abstract-grammar
- http://llvmlite.pydata.org/

About LLVM:

- https://mapping-high-level-constructs-to-llvm-ir.readthedocs.io/
- https://www.llvm.org/docs/tutorial/index.html
- https://llvm.org/docs/LangRef.html

About performance:

- https://llvm.org/docs/Frontend/PerformanceTips.html

About libffi:

- http://www.chiark.greenend.org.uk/doc/libffi-dev/html/index.html

About numpy:

- https://docs.scipy.org/doc/numpy/reference/arrays.interface.html
