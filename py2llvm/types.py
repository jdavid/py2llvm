from llvmlite import ir

try:
    import numpy as np
except ImportError:
    np = None


#
# Basic IR types
#

void = ir.VoidType()
float32 = ir.FloatType()
float64 = ir.DoubleType()
int32 = ir.IntType(32)
int64 = ir.IntType(64)


# Mapping from basic types to IR types
types = {
    float: float64,
    int: int64,
    type(None): void,
}

if np is not None:
    types[np.float32] = float32
    types[np.float64] = float64
    types[np.int32] = int32
    types[np.int32] = int64


def type_to_ir_type(type_):
    """
    Given a Python or IR type, return the corresponding IR type.
    """
    if isinstance(type_, ir.Type):
        return type_

    # None is a special case
    # https://docs.python.org/3/library/typing.html#type-aliases
    if type_ is None:
        return void

    # Basic types
    if type_ in types:
        return types[type_]

    raise ValueError(f'unexpected {type_}')


#
# Compound types
#

class ArrayShape:
    def __init__(self, name):
        self.name = name

class ArrayType:
    def __init__(self, name, ptr):
        self.name = name
        self.ptr = ptr

    @property
    def shape(self):
        return ArrayShape(self.name)

def Array(dtype, ndim):
    return type(
        f'Array[{dtype}, {ndim}]',
        (ArrayType,),
        dict(dtype=dtype, ndim=ndim)
    )


class StructType:

    def __init__(self, name, ptr):
        self.name = name
        self.ptr = ptr

def Struct(name, **kw):
    body = []
    indices = {}
    for i, (k, v) in enumerate(kw.items()):
        v = type_to_ir_type(v)
        body.append(v)
        indices[k] = i

    indices['name'] = name
    indices['body'] = body
    return type(
        f'Struct[{name}, {kw}]',
        (StructType,),
        indices,
    )
