"""
This program translates a small subset of Python to LLVM IR.

Usage:

  import py2llvm as llvm
  def f() -> llvm.double:
    ...

  f = llvm.compile(f)

Notes:

- Only a very small subset is implemented.
- Only integers and floats are supported.
- Integers are translated to 64 bit integers in LLVM, and floats are translated
  to 64 bit doubles in LLVM.
- Function arguments and return values *must* be typed (type hints).

Types. The supported types are float, double, int32 and int64. It's also
possible to use Python's float and int types, they are considered aliases to
double and int64 respectively.

The compiled functions will be called using ctypes, so the allowed
argument types are a subset of the types available in ctypes.

Literals:

- The "2" literal is an integer, the "2.0" literal is a float.
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
"""

# Standard Library
import ast
import builtins
import ctypes
import inspect
import operator
import types
import typing

from llvmlite import ir
from llvmlite import binding

#
# Types and constants
#

float32 = ir.FloatType()
float64 = ir.DoubleType()
int32 = ir.IntType(32)
int64 = ir.IntType(64)

int32p = int32.as_pointer()
int64p = int32.as_pointer()


zero = ir.Constant(int64, 0)
one = ir.Constant(int64, 1)


def value_to_type(value):
    """
    Given a Python or IR value, return its Python or IR type.
    """
    return value.type if isinstance(value, ir.Value) else type(value)


def type_to_ir_type(type_):
    """
    Given a Python or IR type, return the corresponding IR type.
    """
    if isinstance(type_, ir.Type):
        return type_

    # Basic types
    basic_types = {float: float64, int: int64}
    ir_type = basic_types.get(type_)
    if ir_type is not None:
        return ir_type

    # Only List[int] and List[float] are supported
    assert type_.__origin__ is typing.List
    item_type = type_.__args__[0]
    item_type = basic_types[item_type]
    return item_type.as_pointer()


def value_to_ir_type(value):
    """
    Given a Python or IR value, return it's IR type.
    """
    type_ = value_to_type(value)
    return type_to_ir_type(type_)


def values_to_type(left, right):
    """
    Given two values return their type. If mixing Python and IR values, IR
    wins. If mixing integers and floats, float wins.

    If mixing different lengths the longer one wins (e.g. float and double).
    """
    ltype = value_to_type(left)
    rtype = value_to_type(right)

    # Both are Python
    if not isinstance(ltype, ir.Type) and not isinstance(rtype, ir.Type):
        if ltype is float or rtype is float:
            return float

        return int

    # At least 1 is IR
    ltype = type_to_ir_type(ltype)
    rtype = type_to_ir_type(ltype)

    if ltype is float64 or rtype is float64:
        return float64

    if ltype is float32 or rtype is float32:
        return float32

    if ltype is int64 or rtype is int64:
        return int64

    return int32


def value_to_ir_value(value):
    """
    If value is already an IR value just return it. If it's a Python value then
    convert to an IR constant and return it.
    """
    if isinstance(value, ir.Value):
        return value

    ir_type = value_to_ir_type(value)
    return ir.Constant(ir_type, value)



def get_c_type(ir_type):
    basic_types = {
        float32: ctypes.c_float,
        float64: ctypes.c_double,
        int32: ctypes.c_int32,
        int64: ctypes.c_int64,
    }

    c_type = basic_types.get(ir_type)
    if c_type is not None:
        return c_type

    assert ir_type.is_pointer
    return ctypes.POINTER(basic_types[ir_type.pointee])


#
# AST
#

LEAFS = {
    ast.Name, ast.Num,
    # boolop
    ast.And, ast.Or,
    # operator
    ast.Add, ast.Sub, ast.Mult, ast.MatMult, ast.Div, ast.Mod, ast.Pow,
    ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.FloorDiv,
    # unaryop
    ast.Invert, ast.Not, ast.UAdd, ast.USub,
    # cmpop
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
    ast.In, ast.NotIn,
    # expr_context
    ast.Load, ast.Store, ast.Del, ast.AugLoad, ast.AugStore, ast.Param,
}

class BaseNodeVisitor:
    """
    The ast.NodeVisitor class traverses the AST and calls user defined
    callbacks when entering a node.

    Here we do the same thing but we've more callbacks:

    - Callback as well when exiting the node
    - Callback as well after traversing an attribute
    - Except leaf nodes, which are called only once (like in ast.NodeVisitor)
    - To find out the callback we use the MRO

    And we pass more information to the callbacks:

    - Pass the parent node to the callback
    - Pass the value of the attribute to the attribute callback
    - Pass the values of all the attributes to the exit callback

    Override this class and define the callbacks you need:

    - def <classname>_enter(node, parent)
    - def <classname>_<attribute>(node, parent, value)
    - def <classname>_exit(node, parent, *args)

    For leaf nodes use:

    - def <classname>_visit(node, parent)

    Call using traverse:

        class NodeVisitor(BaseNodeVisitor):
            ...

        node = ast.parse(source)
        NodeVisitor().traverse(node)
    """

    def __init__(self, debug):
        self.debug = debug
        self.depth = 0

    @classmethod
    def get_fields(cls, node):
        fields = {
            # returns before body (and after args)
            ast.FunctionDef: ('name', 'args', 'returns', 'body', 'decorator_list'),
            # Do not traverse ctx
            ast.Name: (),
        }

        return fields.get(type(node), node._fields)

    @classmethod
    def iter_fields(cls, node):
        for field in cls.get_fields(node):
            try:
                yield field, getattr(node, field)
            except AttributeError:
                pass

    def traverse(self, node, parent=None):
        if node.__class__ in LEAFS:
            return self.callback('visit', node, parent)

        # Enter
        # enter callback return False to skip traversing the subtree
        if self.callback('enter', node, parent) is False:
            return None

        self.depth += 1

        # Traverse
        args = []
        for name, field in self.iter_fields(node):
            if isinstance(field, list):
                value = [self.traverse(x, node) for x in field if isinstance(x, ast.AST)]
            elif isinstance(field, ast.AST):
                value = self.traverse(field, node)
            else:
                value = field

            self.callback(name, node, parent, value)
            args.append(value)

        # Exit
        self.depth -= 1
        return self.callback('exit', node, parent, *args)

    def callback(self, event, node, parent, *args):
        for cls in node.__class__.__mro__:
            method = f'{cls.__name__}_{event}'
            cb = getattr(self, method, None)
            if cb is not None:
                break

        # Call
        value = cb(node, parent, *args) if cb is not None else None

        # Debug
        if self.debug:
            name = node.__class__.__name__
            line = None
            if event == 'enter':
                line = f'<{name}>'
                if node._fields:
                    attrs = ' '.join(f'{k}' for k, v in ast.iter_fields(node))
                    line = f'<{name} {attrs}>'

                if value is False:
                    line += ' SKIP'
            elif event == 'exit':
                line = f'</{name}> -> {value}'
#               if args:
#                   attrs = ' '.join(repr(x) for x in args)
#                   line = f'</{name} {attrs}>'
#               else:
#                   line = f'</{name}>'
            elif event == 'visit':
                if node._fields:
                    attrs = ' '.join(f'{k}' for k, v in ast.iter_fields(node))
                    line = f'<{name} {attrs} />'
                else:
                    line = f'<{name} />'
                if cb is not None:
                    line += f' -> {value}'
            else:
                if cb is not None:
                    attrs = ' '.join([repr(x) for x in args])
                    line = f'_{event}({attrs})'

            if line:
                print(self.depth * ' ' + line)

        return value


class NodeVisitor(BaseNodeVisitor):

    def lookup(self, name):
        if name in self.locals:
            return self.locals[name]
        if name in self.root.globals:
            return self.root.globals[name]
        return getattr(builtins, name)

    def Module_enter(self, node, parent):
        """
        Module(stmt* body)
        """
        self.root = node

    def arguments_enter(self, node, parent):
        """
        arguments = (arg* args, arg? vararg, arg* kwonlyargs, expr* kw_defaults,
                     arg? kwarg, expr* defaults)
        """
        # We don't parse arguments because arguments are handled in compile
        return False


class BlockVisitor(NodeVisitor):
    """
    The algorithm makes 2 passes to the AST. This is the first one, here:

    - We fail early for features we don't support.
    - We populate the AST attaching structure IR objects (module, functions,
      blocks). These will be used in the 2nd pass.

    In general we should do here as much as we can.
    """

    def Name_visit(self, node, parent):
        """
        Name(identifier id, expr_context ctx)
        """
        name = node.id
        ctx = node.ctx.__class__
        if ctx is ast.Load:
            try:
                return self.lookup(name)
            except AttributeError:
                pass

        return None

    def FunctionDef_enter(self, node, parent):

        """
        FunctionDef(identifier name, arguments args,
                    stmt* body, expr* decorator_list, expr? returns)
        """
        assert type(parent) is ast.Module, 'nested functions not implemented'
        assert not node.decorator_list, 'decorators not implemented'
        assert node.returns is not None, 'expected type-hint for function return type'

        # Initialize function context
        node.locals = {}
        self.locals = node.locals

    def FunctionDef_returns(self, node, parent, returns):
        """
        When we reach this point we have all the function signature: arguments
        and return type.
        """
        root = self.root
        f_name = node.name
        f_args = root.f_args

        # Create the function
        function = root.ir_function
        self.root.globals[f_name] = function

        # Create the first block of the function, and the associated builder.
        # The first block, named "vars", is where all local variables will be
        # allocated. We will keep it open until we close the function in the
        # 2nd pass.
        block_vars = function.append_basic_block('vars')
        builder = ir.IRBuilder(block_vars)

        # Function start: allocate a local variable for every argument
        for i, (name, py_type) in enumerate(f_args):
            arg = function.args[i]
            ptr = builder.alloca(arg.type, name=name)
            builder.store(arg, ptr)
            # Keep Give a name to the arguments, and keep them in local namespace
            node.locals[name] = ptr

        # Create the second block, this is where the code proper will start,
        # after allocation of the local variables.
        block_start = function.append_basic_block('start')
        builder.position_at_end(block_start)

        # Keep stuff we will need in this first pass
        self.function = function

        # Keep stuff for the second pass
        node.block_vars = block_vars
        node.block_start = block_start
        node.builder = builder
        node.f_rtype = root.f_returns

    def If_test(self, node, parent, test):
        """
        If(expr test, stmt* body, stmt* orelse)
        """
        node.block_true = self.function.append_basic_block()

    def If_body(self, node, parent, body):
        node.block_false = self.function.append_basic_block()

    def If_orelse(self, node, parent, orelse):
        node.block_next = self.function.append_basic_block()

    def For_enter(self, node, parent):
        """
        For(expr target, expr iter, stmt* body, stmt* orelse)
        """
        assert not node.orelse, '"for ... else .." not supported'
        node.block_for = self.function.append_basic_block('for')
        node.block_body = self.function.append_basic_block('for_body')

    def For_exit(self, node, parent, *args):
        node.block_next = self.function.append_basic_block('for_out')

    def While_enter(self, node, parent):
        """
        While(expr test, stmt* body, stmt* orelse)
        """
        assert not node.orelse, '"while ... else .." not supported'
        node.block_while = self.function.append_basic_block('while')
        node.block_body = self.function.append_basic_block('while_body')

    def While_exit(self, node, parent, *args):
        node.block_next = self.function.append_basic_block('while_out')


class GenVisitor(NodeVisitor):
    """
    Builtin types are:
    identifier, int, string, bytes, object, singleton, constant

    singleton: None, True or False
    constant can be None, whereas None means "no value" for object.
    """

    module = None
    function = None
    args = None
    builder = None
    f_rtype = None # Type of the return value
    ltype = None # Type of the local variable

    def print(self, line):
        print(self.depth * ' ' + line)

    def debug(self, node, parent):
        for name, field in ast.iter_fields(node):
            self.print(f'- {name} {field}')

    def convert(self, value, type_):
        """
        Return the value converted to the given type.
        """
        # If Python value, return a constant
        if not isinstance(value, ir.Value):
            return ir.Constant(type_, value)

        if value.type is type_:
            return value

        conversions = {
            (ir.IntType, ir.FloatType): self.builder.sitofp,
            (ir.FloatType, ir.IntType): self.builder.fptosi,
            (ir.IntType, ir.DoubleType): self.builder.sitofp,
            (ir.DoubleType, ir.IntType): self.builder.fptosi,
        }
        conversion = conversions.get((type(value.type), type(type_)))
        if conversion is None:
            err = f'Conversion from {value.type} to {type_} not suppoerted'
            raise NotImplementedError(err)

        return conversion(value, type_)

    #
    # Leaf nodes
    #

    def Name_visit(self, node, parent):
        """
        Name(identifier id, expr_context ctx)
        """
        name = node.id
        ctx = node.ctx.__class__

        if ctx is ast.Load:
            value = self.lookup(name)
            if type(value) is ir.AllocaInstr:
                if not isinstance(value.type.pointee, ir.Aggregate):
                    return self.builder.load(value)
            return value

        if ctx is ast.Store:
            return name
        raise NotImplementedError(f'unexpected ctx={ctx}')

    def Num_visit(self, node, parent):
        """
        Num(object n)
        """
        return node.n

    def boolop_visit(self, node, parent):
        return type(node)

    def operator_visit(self, node, parent):
        return type(node)

    def unaryop_visit(self, node, parent):
        return type(node)

    def Eq_visit(self, node, parent):
        return '=='

    def NotEq_visit(self, node, parent):
        return '!='

    def Lt_visit(self, node, parent):
        return '<'

    def LtE_visit(self, node, parent):
        return '<='

    def Gt_visit(self, node, parent):
        return '>'

    def GtE_visit(self, node, parent):
        return '>='

    #
    # Literals
    #

    def List_exit(self, node, parent, elts, ctx):
        """
        List(expr* elts, expr_context ctx)
        """
        py_types = {type(x) for x in elts}
        n = len(py_types)
        if n == 0:
            py_type = int # any type will do because the list is empty
        elif n == 1:
            py_type = py_types.pop()
        else:
            raise TypeError('all list elements must be of the same type')

        el_type = type_to_ir_type(py_type)
        typ = ir.ArrayType(el_type, len(elts))
        return ir.Constant(typ, elts)

    #
    # Expressions
    #

    def FunctionDef_enter(self, node, parent):
        self.locals = node.locals
        self.builder = node.builder
        self.f_rtype = node.f_rtype
        self.block_vars = node.block_vars

    def FunctionDef_exit(self, node, parent, *args):
        node.builder.position_at_end(node.block_vars)
        node.builder.branch(node.block_start)

    def BinOp_exit(self, node, parent, left, op, right):
        type_ = values_to_type(left, right)

        # Two Python values
        if not isinstance(type_, ir.Type):
            ast2op = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
            }
            py_op = ast2op.get(op)
            if py_op is None:
                raise NotImplementedError(
                    f'{op.__name__} operator for {type_} type not implemented')
            return py_op(left, right)

        # One or more IR values
        left = self.convert(left, type_)
        right = self.convert(right, type_)

        d = {
            (ast.Add,  ir.IntType): self.builder.add,
            (ast.Sub,  ir.IntType): self.builder.sub,
            (ast.Mult, ir.IntType): self.builder.mul,
            (ast.Div,  ir.IntType): self.builder.sdiv,
            (ast.Add,  ir.FloatType): self.builder.fadd,
            (ast.Sub,  ir.FloatType): self.builder.fsub,
            (ast.Mult, ir.FloatType): self.builder.fmul,
            (ast.Div,  ir.FloatType): self.builder.fdiv,
            (ast.Add,  ir.DoubleType): self.builder.fadd,
            (ast.Sub,  ir.DoubleType): self.builder.fsub,
            (ast.Mult, ir.DoubleType): self.builder.fmul,
            (ast.Div,  ir.DoubleType): self.builder.fdiv,
        }
        base_type = type(type_)
        ir_op = d.get((op, base_type))
        if ir_op is None:
            raise NotImplementedError(
                f'{op.__name__} operator for {type_} type not implemented')

        return ir_op(left, right)

    def Compare_exit(self, node, parent, left, ops, comparators):
        """
        Compare(expr left, cmpop* ops, expr* comparators)
        """
        assert len(ops) == 1
        assert len(comparators) == 1
        op = ops[0]
        right = comparators[0]

        type_ = values_to_type(left, right)

        # Two Python values
        if not isinstance(type_, ir.Type):
            ast2op = {
                '==': operator.eq,
                '!=': operator.ne,
                '<': operator.lt,
                '<=': operator.le,
                '>': operator.gt,
                '>=': operator.ge,
            }
            py_op = ast2op.get(op)
            return py_op(left, right)

        # At least 1 IR value
        left = self.convert(left, type_)
        right = self.convert(right, type_)

        d = {
            ir.IntType: self.builder.icmp_signed,
            ir.FloatType: self.builder.fcmp_unordered, # XXX fcmp_ordered
            ir.DoubleType: self.builder.fcmp_unordered, # XXX fcmp_ordered
        }
        base_type = type(type_)
        ir_op = d.get(base_type)
        return ir_op(op, left, right)

    def BoolOp_exit(self, node, parent, op, values):
        """
        BoolOp(boolop op, expr* values)
        """
        ir_op = {
            ast.And: self.builder.and_,
            ast.Or: self.builder.or_,
        }[op]

        assert len(values) == 2
        left, right = values
        return ir_op(left, right)

    def UnaryOp_exit(self, node, parent, op, operand):
        """
        UnaryOp(unaryop op, expr operand)
        """
        ir_op = {
            ast.Not: self.builder.not_,
        }[op]

        return ir_op(operand)

    def Index_exit(self, node, parent, value):
        """
        Index(expr value)
        """
        return value

    def Subscript_exit(self, node, parent, value, slice, ctx):
        """
        Subscript(expr value, slice slice, expr_context ctx)
        """
        idx = value_to_ir_value(slice)
        assert value.type.is_pointer
        pointee = value.type.pointee
        if isinstance(pointee, ir.ArrayType):
            ptr = self.builder.gep(value, [zero, idx])
        else:
            ptr = self.builder.gep(value, [idx])
        return self.builder.load(ptr)

    #
    # if .. elif .. else
    #
    def If_test(self, node, parent, test):
        """
        If(expr test, stmt* body, stmt* orelse)
        """
        self.builder.cbranch(test, node.block_true, node.block_false)
        self.builder.position_at_end(node.block_true)

    def If_body(self, node, parent, body):
        if not self.builder.block.is_terminated:
            self.builder.branch(node.block_next)
        self.builder.position_at_end(node.block_false)

    def If_orelse(self, node, parent, orelse):
        self.builder.branch(node.block_next)
        self.builder.position_at_end(node.block_next)

    #
    # for ...
    #
    def For_iter(self, node, parent, expr):
        node.i = self.builder.alloca(int64, name='i')       # i
        self.builder.store(zero, node.i)                    # i = 0
        arr = self.builder.alloca(expr.type)                # arr
        self.builder.store(expr, arr)                       # arr = expr
        n = ir.Constant(int64, expr.type.count)             # n = len(expr)
        self.builder.branch(node.block_for)                 # br %for

        self.builder.position_at_end(node.block_for)         # %for
        idx = self.builder.load(node.i)                      # %idx = i
        test = self.builder.icmp_unsigned('<', idx, n)       # %idx < n
        self.builder.cbranch(test, node.block_body, node.block_next) # br %test %body %next

        self.builder.position_at_end(node.block_body)        # %body
        ptr = self.builder.gep(arr, [zero, idx])             # expr[idx]
        x = self.builder.load(ptr)                           # % = expr[i]
        self.locals[node.target.id] = x

    def For_exit(self, node, parent, *args):
        a = self.builder.load(node.i)                        # % = i
        b = self.builder.add(a, one)                         # % = % + 1
        self.builder.store(b, node.i)                        # i = %
        self.builder.branch(node.block_for)                  # br %for
        self.builder.position_at_end(node.block_next)        # %next
    #
    # while ...
    #
    def While_enter(self, node, parent):
        self.builder.branch(node.block_while)
        self.builder.position_at_end(node.block_while)

    def While_test(self, node, parent, test):
        self.builder.cbranch(test, node.block_body, node.block_next)
        self.builder.position_at_end(node.block_body)

    def While_exit(self, node, parent, *args):
        self.builder.branch(node.block_while)
        self.builder.position_at_end(node.block_next)

    #
    # Other non-leaf nodes
    #
    def AnnAssign_annotation(self, node, parent, value):
        self.ltype = value

    def AnnAssign_exit(self, node, parent, target, annotation, value, simple):
        """
        AnnAssign(expr target, expr annotation, expr? value, int simple)
        """
        assert value is not None
        assert simple == 1

        ltype = type_to_ir_type(self.ltype)
        value = self.convert(value, ltype)
        self.ltype = None

        name = target
        try:
            ptr = self.lookup(name)
        except AttributeError:
            block_cur = self.builder.block
            self.builder.position_at_end(self.block_vars)
            ptr = self.builder.alloca(value.type, name=name)
            self.builder.position_at_end(block_cur)
            self.locals[name] = ptr

        return self.builder.store(value, ptr)

    def Assign_enter(self, node, parent):
        """
        Assign(expr* targets, expr value)
        """
        assert len(node.targets) == 1
        assert type(node.targets[0]) is ast.Name
        assert type(node.targets[0].ctx) is ast.Store

    def Assign_exit(self, node, parent, name, value):
        name = name[0]
        value = value_to_ir_value(value)

        try:
            ptr = self.lookup(name)
        except AttributeError:
            block_cur = self.builder.block
            self.builder.position_at_end(self.block_vars)
            ptr = self.builder.alloca(value.type, name=name)
            self.builder.position_at_end(block_cur)
            self.locals[name] = ptr

        return self.builder.store(value, ptr)

    def Return_enter(self, node, parent):
        self.ltype = self.f_rtype

    def Return_exit(self, node, parent, value):
        """
        Return(expr? value)
        """
        value = self.convert(value, self.f_rtype)
        self.ltype = None
        return self.builder.ret(value)

    def Call_exit(self, node, parent, func, args, keywords):
        """
        Call(expr func, expr* args, keyword* keywords)
        """
        assert not keywords
        return self.builder.call(func, args)


class LLVMFunction:
    """
    Wraps a Python function. Compiled to IR, it can be executed with ctypes:

    f(...)

    Besides calling the function a number of attributes are available:

    name        -- the name of the function
    py_function -- the original Python function
    py_source   -- the source code of the Python function
    ir          -- LLVM's IR code
    cfunctype   -- the C signature, used when calling
    cfunction   -- the C function (only this one is needed to make the call)
    """

    def __init__(self, c_args, func_ptr, name,
                 py_function=None, py_source=None, ir=None):

        # Get the function
        self.c_args = c_args
        self.cfunctype = ctypes.CFUNCTYPE(*c_args)
        self.cfunction = self.cfunctype(func_ptr)

        # Keep stuff for introspection
        self.name = name
        # Optional stuff
        self.py_function = py_function
        self.py_source = py_source
        self.ir = ir

    def __call__(self, *args, debug=False):
        c_args = []
        for i, arg in enumerate(args):
            if type(arg) is list:
                c_arg = self.c_args[i+1]
                arg = (c_arg._type_ * len(arg))(*arg)
            c_args.append(arg)

        value = self.cfunction(*c_args)
        if debug:
            print(f'{args} => {value}')

        return value


class LLVM:

    def __init__(self):
        self.engine = self.create_execution_engine()

    def compile(self, py_function, verbose=0):
        assert type(py_function) is types.FunctionType

        # We only support positional arguments
        signature = inspect.signature(py_function)
        params = signature.parameters
        params = [params[x] for x in params]
        for param in params:
            if param.kind > inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise NotImplementedError('only positional arguments are supported')

        # This will be the final output of the whole process
        ir_module = ir.Module()

        # The IR signature
        returns = type_to_ir_type(signature.return_annotation)
        params = [
            (param.name, type_to_ir_type(param.annotation))
            for param in params
        ]

        f_type = ir.FunctionType(returns, tuple(type_ for name, type_ in params))
        f_name = py_function.__name__
        ir_function = ir.Function(ir_module, f_type, f_name)

        # The ctypes function signature is needed to call the compiled function
        c_args = [returns] + [py_type for name, py_type in params]
        c_args = [get_c_type(t) for t in c_args]

        # Python AST
        py_source = inspect.getsource(py_function)
        if verbose:
            print('====== Source ======')
            print(py_source)

        node = ast.parse(py_source)
        node.globals = inspect.stack()[1].frame.f_globals
        node.ir_function = ir_function
        node.f_args = params
        node.f_returns = returns

        # AST 1st pass: structure
        debug = verbose > 1
        if debug: print('====== Debug: 1st pass ======')
        BlockVisitor(debug).traverse(node)

        # AST 2nd pass: generate
        if debug: print('====== Debug: 2nd pass ======')
        GenVisitor(debug).traverse(node)

        # IR code
        ir_source = str(ir_module)
        if verbose:
            print('====== IR ======')
            print(ir_source)
        self.compile_ir(ir_source) # Compile

        # Return the function wrapper
        func_ptr = self.engine.get_function_address(f_name)
        return LLVMFunction(c_args, func_ptr, f_name, py_function,
                            py_source, ir_source)

    def create_execution_engine(self):
        """
        Create an ExecutionEngine suitable for JIT code generation on
        the host CPU.  The engine is reusable for an arbitrary number of
        modules.
        """
        # Create a target machine representing the host
        target = binding.Target.from_default_triple()
        target_machine = target.create_target_machine()
        # And an execution engine with an empty backing module
        backing_mod = binding.parse_assembly("")
        engine = binding.create_mcjit_compiler(backing_mod, target_machine)
        return engine

    def compile_ir(self, llvm_ir):
        """
        Compile the LLVM IR string with the given engine.
        The compiled module object is returned.
        """
        engine = self.engine

        # Create a LLVM module object from the IR
        mod = binding.parse_assembly(llvm_ir)
        mod.verify()
        # Now add the module and make sure it is ready for execution
        engine.add_module(mod)
        engine.finalize_object()
        engine.run_static_constructors()
        return mod


# All these initializations are required for code generation!
binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()  # yes, even this one

llvm = LLVM()


#
# Public interface
#

compile = llvm.compile

__all__ = ['compile', 'float32', 'float64', 'int32', 'int64']
