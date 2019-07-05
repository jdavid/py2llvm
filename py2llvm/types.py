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
int8 = ir.IntType(8)
int16 = ir.IntType(16)
int32 = ir.IntType(32)
int64 = ir.IntType(64)

# Pointers
int8p = int8.as_pointer()

# Constants
zero32 = ir.Constant(int32, 0)

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

    def subscript(self, visitor, slice):
        value = visitor.lookup(f'{self.name}_{slice}')
        return visitor.builder.load(value)


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

    @classmethod
    def get_body(self):
        return [type_to_ir_type(type_) for name, type_ in self._fields_]

    def get_index(self, field_name):
        for i, (name, type_) in enumerate(self._fields_):
            if field_name == name:
                return i

        return None

    def __getattr__(self, attr):
        i = self.get_index(attr)
        if i is None:
            raise AttributeError(f'Unexpected {attr}')

        def cb(visitor):
            idx = ir.Constant(int32, i)
            ptr = visitor.builder.load(self.ptr)
            ptr = visitor.builder.gep(ptr, [zero32, idx])
            return visitor.builder.load(ptr)

        return cb


def Struct(name, **kw):
    type_dict = {
        '_name_': name,
        '_fields_': kw.items(),
    }
    return type(
        f'Struct[{name}, {kw}]',
        (StructType,),
        type_dict,
    )
