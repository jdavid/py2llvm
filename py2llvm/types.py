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
int64p = int64.as_pointer()

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


def value_to_ir_value(builder, value, type_=None):
    """
    Return a IR value for the given value, where value may be either a Python
    or an IR value. If type_ is given the value will be converted to the given
    type (if necessary).
    """
    if type_ is None:
        type_ = value_to_ir_type(value)

    # If Python value, return a constant
    if not isinstance(value, ir.Value):
        return ir.Constant(type_, value)

    if value.type is type_:
        return value

    conversions = {
        # Integer to float
        (ir.IntType, ir.FloatType): builder.sitofp,
        (ir.IntType, ir.DoubleType): builder.sitofp,
        # Float to integer
        (ir.FloatType, ir.IntType): builder.fptosi,
        (ir.DoubleType, ir.IntType): builder.fptosi,
        # Float to float
        (ir.FloatType, ir.DoubleType): builder.fpext,
        (ir.DoubleType, ir.FloatType): builder.fptrunc,
    }

    if isinstance(value.type, ir.IntType) and isinstance(type_, ir.IntType):
        # Integer to integer
        if value.type.width < type_.width:
            conversion = builder.zext
        else:
            conversion = builder.trunc
    else:
        # To or from float
        conversion = conversions.get((type(value.type), type(type_)))
        if conversion is None:
            err = f'Conversion from {value.type} to {type_} not supported'
            raise NotImplementedError(err)

    return conversion(value, type_)


#
# Compound types
#

class ComplexType:

    def __init__(self, function, name, args):
        self.name = name
        self.ptr = args[name]

    def preamble(self, builder):
        pass


class ArrayShape:

    def __init__(self, shape):
        self.shape = shape

    def get(self, visitor, n):
        value = self.shape[n]
        return visitor.builder.load(value)

    def subscript(self, visitor, slice, ctx):
        assert ctx is ast.Load
        return self.get(visitor, slice)


class ArrayType(ComplexType):

    def __init__(self, function, name, args):
        super().__init__(function, name, args)
        # Keep a pointer to every dimension
        prefix = f'{name}_'
        n = len(prefix)
        shape = {int(x[n:]): args[x] for x in args if x.startswith(prefix)}
        self.shape = ArrayShape(shape)

    def get_ptr(self, visitor):
        return visitor.builder.load(self.ptr)

    def subscript(self, visitor, slice, ctx):
        builder = visitor.builder

        # To make it simpler, make the slice to be a list always
        if type(slice) is not list:
            slice = [slice]

        # Get the pointer to the beginning
        ptr = self.get_ptr(visitor)

        assert ptr.type.is_pointer
        if isinstance(ptr.type.pointee, ir.ArrayType):
            ptr = builder.gep(ptr, [zero])

        # Support for multidimensional arrays.
        # Let's we have 3 dimensions (d0, d1, d2), each with a length (dl0,
        # dl1, dl2). Then the size of the dimensions (ds0, ds1, ds2) is
        # calculated multiplying the length of the next dimensions, for
        # example: ds0 = dl1 * dl2
        # Because we assume the array is stored using the C convention.
        dim = 1
        while slice:
            idx = slice.pop(0)
            idx = value_to_ir_value(builder, idx)
            for i in range(dim, self.ndim):
                dim_len = self.shape.get(visitor, dim)
                idx = builder.mul(idx, dim_len)

            ptr = builder.gep(ptr, [idx])
            dim += 1

        # Return the value
        if ctx is ast.Load:
            return builder.load(ptr)
        elif ctx is ast.Store:
            return ptr


def Array(dtype, ndim):
    return type(
        f'Array[{dtype}, {ndim}]',
        (ArrayType,),
        dict(dtype=dtype, ndim=ndim)
    )


class Node:

    def Attribute_exit(self, visitor):
        return self


class StructAttrNode(Node):

    def __init__(self, ptr, i):
        self.ptr = ptr
        self.i = i

    def Attribute_exit(self, visitor):
        idx = ir.Constant(int32, self.i)
        ptr = visitor.builder.load(self.ptr)
        ptr = visitor.builder.gep(ptr, [zero32, idx])
        return ptr


class StructType(ComplexType):

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

        return StructAttrNode(self.ptr, i)


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
