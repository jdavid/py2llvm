import ast

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
zero = ir.Constant(int64, 0)
one = ir.Constant(int64, 1)
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


def value_to_type(value):
    """
    Given a Python or IR value, return its Python or IR type.
    """
    return value.type if isinstance(value, ir.Value) else type(value)


def value_to_ir_type(value):
    """
    Given a Python or IR value, return it's IR type.
    """
    if np is not None and isinstance(value, np.ndarray):
        return Array(value.dtype.type, value.ndim)

    type_ = value_to_type(value)
    return type_to_ir_type(type_)


def value_to_ir_value(value, visitor=None):
    """
    Return a IR value for the given value, where value may be:
    - an IR value, then we're done, return it
    - a special object that knows to return a IR value (duck typing)
    - a regular Python value, then return a IR constant
    """
    if isinstance(value, ir.Value):
        return value

    # Special object
    to_ir_value = getattr(value, 'to_ir_value', None)
    if to_ir_value is not None:
        assert visitor is not None
        return to_ir_value(visitor)

    # Regular Python object
    ir_type = value_to_ir_type(value)
    return ir.Constant(ir_type, value)


#
# Compound types
#

class ArrayShape:
    def __init__(self, name):
        self.name = name

    def subscript(self, visitor, slice, ctx):
        assert ctx is ast.Load
        value = visitor.lookup(f'{self.name}_{slice}')
        return visitor.builder.load(value)


class ArrayType:
    def __init__(self, name, ptr):
        self.name = name
        self.ptr = ptr

    @property
    def shape(self):
        return ArrayShape(self.name)

    def subscript(self, visitor, slice, ctx):
        # To make it simpler, make the slice to be a list always
        if type(slice) is not list:
            slice = [slice]

        # Get the pointer to the beginning
        ptr = visitor.builder.load(self.ptr)

        assert ptr.type.is_pointer
        if isinstance(ptr.type.pointee, ir.ArrayType):
            ptr = visitor.builder.gep(ptr, [zero])

        # Support for multidimensional arrays
        dim = 1
        while slice:
            idx = slice.pop(0)
            idx = value_to_ir_value(idx)
            for i in range(dim, self.ndim):
                dim_len = visitor.lookup(f'{self.name}_{dim}')
                dim_len = visitor.builder.load(dim_len)
                idx = visitor.builder.mul(idx, dim_len)

            ptr = visitor.builder.gep(ptr, [idx])
            dim += 1

        # Return the value
        if ctx is ast.Load:
            return visitor.builder.load(ptr)
        elif ctx is ast.Store:
            return ptr


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
            return ptr

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
