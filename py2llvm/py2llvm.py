# Standard Library
import ast
import builtins
import collections
import ctypes
import inspect
import math
import operator
from types import FunctionType
import typing

# Requirements
from llvmlite import binding, ir

from . import types


class Range:

    def __init__(self, *args):
        start = step = None

        # Unpack
        n = len(args)
        if n == 1:
            stop, = args
        elif n == 2:
            start, stop = args
        else:
            start, stop, step = args

        # Defaults
        type_ = stop.type if isinstance(stop, ir.Value) else types.int64
        if start is None:
            start = ir.Constant(type_, 0)
        if step is None:
            step = ir.Constant(type_, 1)

        # Keep IR values
        self.start = types.value_to_ir_value(start)
        self.stop = types.value_to_ir_value(stop)
        self.step = types.value_to_ir_value(step)


def values_to_type(left, right):
    """
    Given two values return their type. If mixing Python and IR values, IR
    wins. If mixing integers and floats, float wins.

    If mixing different lengths the longer one wins (e.g. float and double).
    """
    ltype = types.value_to_type(left)
    rtype = types.value_to_type(right)

    # Both are Python
    if not isinstance(ltype, ir.Type) and not isinstance(rtype, ir.Type):
        if ltype is float or rtype is float:
            return float

        return int

    # At least 1 is IR
    ltype = types.type_to_ir_type(ltype)
    rtype = types.type_to_ir_type(ltype)

    if ltype is types.float64 or rtype is types.float64:
        return types.float64

    if ltype is types.float32 or rtype is types.float32:
        return types.float32

    if ltype is types.int64 or rtype is types.int64:
        return types.int64

    return types.int32


#
# AST
#

LEAFS = {
    ast.Name, ast.NameConstant, ast.Num,
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

    def __init__(self, verbose):
        self.verbose = verbose
        self.depth = 0

    @classmethod
    def get_fields(cls, node):
        fields = {
            # Skip "decorator_list", and traverse "returns" before "body"
            # ('name', 'args', 'body', 'decorator_list', 'returns')
            ast.FunctionDef: ('name', 'args', 'returns', 'body'),
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
        if self.verbose > 1:
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

        # To support recursivity XXX
        if name in self.root.compiled:
            return self.root.compiled[name]

        if name in self.root.globals:
            return self.root.globals[name]

        return getattr(builtins, name)

    def Module_enter(self, node, parent):
        """
        Module(stmt* body)
        """
        self.root = node

    def FunctionDef_enter(self, node, parent):
        """
        FunctionDef(identifier name, arguments args,
                    stmt* body, expr* decorator_list, expr? returns)
        """
        assert type(parent) is ast.Module, 'nested functions not implemented'

        # Initialize function context
        node.locals = {}
        self.locals = node.locals

    def arguments_enter(self, node, parent):
        """
        arguments = (arg* args, arg? vararg, arg* kwonlyargs, expr* kw_defaults,
                     arg? kwarg, expr* defaults)
        """
        # We don't parse arguments because arguments are handled in compile
        return False

    def Assign_enter(self, node, parent):
        """
        Assign(expr* targets, expr value)
        """
        assert len(node.targets) == 1, 'Unpacking not supported'

    #
    # Leaf nodes
    #
    def NameConstant_visit(self, node, parent):
        """
        NameConstant(singleton value)
        """
        return node.value

    def Num_visit(self, node, parent):
        """
        Num(object n)
        """
        return node.n

    def expr_context_visit(self, node, parent):
        return type(node)

    def Name_visit(self, node, parent):
        """
        Name(identifier id, expr_context ctx)
        """
        name = node.id
        ctx = type(node.ctx)

        if ctx is ast.Load:
            try:
                return self.lookup(name)
            except AttributeError:
                return None

        elif ctx is ast.Store:
            return name

        raise NotImplementedError(f'unexpected ctx={ctx}')


class InferVisitor(NodeVisitor):
    """
    This optional pass is to infer the return type of the function if not given
    explicitely.
    """

    def Assign_exit(self, node, parent, targets, value):
        target = targets[0]
        if type(target) is str:
            # x =
            self.locals.setdefault(target, value)

    def Return_exit(self, node, parent, value):
        return_type = type(value)

        root = self.root
        if root.return_type is inspect._empty:
            root.return_type = return_type
            return

        assert root.return_type is return_type

    def FunctionDef_exit(self, node, parent, *args):
        root = self.root
        if root.return_type is inspect._empty:
            root.return_type = None


class BlockVisitor(NodeVisitor):
    """
    The algorithm makes 2 passes to the AST. This is the first one, here:

    - We fail early for features we don't support.
    - We populate the AST attaching structure IR objects (module, functions,
      blocks). These will be used in the 2nd pass.
    """

    def FunctionDef_returns(self, node, parent, returns):
        """
        When we reach this point we have all the function signature: arguments
        and return type.
        """
        root = self.root
        ir_signature = root.ir_signature

        # Keep the function in globals so it can be called
        function = root.ir_function
        self.root.compiled[node.name] = function

        # Create the first block of the function, and the associated builder.
        # The first block, named "vars", is where all local variables will be
        # allocated. We will keep it open until we close the function in the
        # 2nd pass.
        block_vars = function.append_basic_block('vars')
        builder = ir.IRBuilder(block_vars)

        # Function start: allocate a local variable for every argument
        for i, param in enumerate(ir_signature.parameters):
            arg = function.args[i]
            assert arg.type is param.type
            ptr = builder.alloca(arg.type, name=param.name)
            builder.store(arg, ptr)
            # Keep Give a name to the arguments, and keep them in local namespace
            node.locals[param.name] = ptr

        # Keep an ArrayType instance, actually, so we can resolve .shape[idx]
        for param in root.py_signature.parameters:
            if type(param.type) is type and issubclass(param.type, types.ComplexType):
                ptr = node.locals.pop(param.name)
                var = param.type(param.name, ptr)
                node.locals.update(var.get_locals())

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
        node.f_rtype = ir_signature.return_type

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
            # Integer to float
            (ir.IntType, ir.FloatType): self.builder.sitofp,
            (ir.IntType, ir.DoubleType): self.builder.sitofp,
            # Float to integer
            (ir.FloatType, ir.IntType): self.builder.fptosi,
            (ir.DoubleType, ir.IntType): self.builder.fptosi,
            # Float to float
            (ir.FloatType, ir.DoubleType): self.builder.fpext,
            (ir.DoubleType, ir.FloatType): self.builder.fptrunc,
        }

        if isinstance(value.type, ir.IntType) and isinstance(type_, ir.IntType):
            # Integer to integer
            if value.type.width < type_.width:
                conversion = self.builder.zext
            else:
                conversion = self.builder.trunc
        else:
            # To or from float
            conversion = conversions.get((type(value.type), type(type_)))
            if conversion is None:
                err = f'Conversion from {value.type} to {type_} not supported'
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
        ctx = type(node.ctx)

        if ctx is ast.Load:
            value = self.lookup(name)
            if type(value) is ir.AllocaInstr:
                if not isinstance(value.type.pointee, ir.Aggregate):
                    return self.builder.load(value)
            return value

        elif ctx is ast.Store:
            return name

        raise NotImplementedError(f'unexpected ctx={ctx}')

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

        el_type = types.type_to_ir_type(py_type)
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
        if self.root.py_signature.return_type is None:
            if not self.builder.block.is_terminated:
                node.builder.ret_void()

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
        # An smart object
        subscript = getattr(value, 'subscript', None)
        if subscript is not None:
            return subscript(self, slice, ctx)

        # A pointer!
        if isinstance(value, ir.Value) and value.type.is_pointer:
            ptr = value
            ptr = self.builder.gep(ptr, [slice])
            return self.builder.load(ptr)

        raise NotImplementedError(f'{type(value)} does not support subscript []')

    def Tuple_exit(self, node, parent, elts, ctx):
        """
        Tuple(expr* elts, expr_context ctx)
        """
        assert ctx is ast.Load
        return elts

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
        """
        For(expr target, expr iter, stmt* body, stmt* orelse)
        """
        target = node.target.id
        if isinstance(expr, Range):
            start = expr.start
            stop = expr.stop
            node.step = expr.step
            name = target
        else:
            start = types.zero
            stop = ir.Constant(types.int64, expr.type.count)
            node.step = types.one
            name = 'i'
            # Allocate and store the literal array to iterate
            arr = self.builder.alloca(expr.type)
            self.builder.store(expr, arr)

        # Allocate and initialize the index variable
        node.i = self.builder.alloca(stop.type, name=name)
        self.builder.store(start, node.i)                         # i = start
        self.builder.branch(node.block_for)                       # br %for

        # Stop condition
        self.builder.position_at_end(node.block_for)              # %for
        idx = self.builder.load(node.i)                           # %idx = i
        test = self.builder.icmp_unsigned('<', idx, stop)         # %idx < stop
        self.builder.cbranch(test, node.block_body, node.block_next) # br %test %body %next
        self.builder.position_at_end(node.block_body)             # %body

        # Keep variable to use within the loop
        if isinstance(expr, Range):
            self.locals[target] = idx
        else:
            ptr = self.builder.gep(arr, [types.zero, idx])        # expr[idx]
            x = self.builder.load(ptr)                            # % = expr[i]
            self.locals[target] = x

    def For_exit(self, node, parent, *args):
        # Increment index variable
        a = self.builder.load(node.i)                             # % = i
        b = self.builder.add(a, node.step)                        # % = % + step
        self.builder.store(b, node.i)                             # i = %
        # Continue
        self.builder.branch(node.block_for)                       # br %for
        self.builder.position_at_end(node.block_next)             # %next

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
    def Attribute_exit(self, node, parent, value, attr, ctx):
        """
        Attribute(expr value, identifier attr, expr_context ctx)
        """
        assert ctx is ast.Load
        value = getattr(value, attr)
        if isinstance(value, types.Node):
            value = value.Attribute_exit(self)

        if isinstance(value, ir.Value) and value.type.is_pointer:
            value = self.builder.load(value)

        return value

    def AnnAssign_annotation(self, node, parent, value):
        self.ltype = value

    def AnnAssign_exit(self, node, parent, target, annotation, value, simple):
        """
        AnnAssign(expr target, expr annotation, expr? value, int simple)
        """
        assert value is not None
        assert simple == 1

        ltype = types.type_to_ir_type(self.ltype)
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

    def Assign_exit(self, node, parent, targets, value):
        if len(targets) > 1:
            raise NotImplementedError(f'unpacking not supported')

        value = types.value_to_ir_value(value, visitor=self)

        target = targets[0]
        if type(target) is str:
            # x =
            name = target
            try:
                ptr = self.lookup(name)
            except AttributeError:
                block_cur = self.builder.block
                self.builder.position_at_end(self.block_vars)
                ptr = self.builder.alloca(value.type, name=name)
                self.builder.position_at_end(block_cur)
                self.locals[name] = ptr
        else:
            # x[i] =
            ptr = target

        return self.builder.store(value, ptr)

    def Return_enter(self, node, parent):
        self.ltype = self.f_rtype

    def Return_exit(self, node, parent, value):
        """
        Return(expr? value)
        """
        if value is None:
            assert self.f_rtype is types.void
            return self.builder.ret_void()

        value = self.convert(value, self.f_rtype)
        self.ltype = None
        return self.builder.ret(value)

    def Call_exit(self, node, parent, func, args, keywords):
        """
        Call(expr func, expr* args, keyword* keywords)
        """
        assert not keywords
        if func is range:
            return Range(*args)

        func = self.root.compiled.get(func, func)
        return self.builder.call(func, args)


Parameter = collections.namedtuple('Parameter', ['name', 'type'])

class Signature:
    def __init__(self, parameters, return_type):
        self.parameters = parameters
        self.return_type = return_type

class Function:
    """
    Wraps a Python function. Compiled to IR, it will be executed with libffi:

    f(...)

    Besides calling the function a number of attributes are available:

    name        -- the name of the function
    py_function -- the original Python function
    py_source   -- the source code of the Python function
    ir          -- LLVM's IR code
    """

    def __init__(self, llvm, py_function, signature, f_globals, optimize=False):
        assert type(py_function) is FunctionType
        self.llvm = llvm
        self.py_function = py_function
        self.name = py_function.__name__

        self.py_signature = self.get_py_signature(signature)
        self.compiled = False
        self.globals = f_globals
        self.optimize = optimize

    def get_py_signature(self, signature):
        inspect_signature = inspect.signature(self.py_function)
        if signature is not None:
            assert len(signature) == len(inspect_signature.parameters) + 1

        # Parameters
        params = []
        for i, name in enumerate(inspect_signature.parameters):
            param = inspect_signature.parameters[name]
            assert param.kind <= inspect.Parameter.POSITIONAL_OR_KEYWORD, \
                   'only positional arguments are supported'

            type_ = param.annotation if signature is None else signature[i]
            params.append(Parameter(name, type_))

        # The return type
        if signature is None:
            return_type = inspect_signature.return_annotation
        else:
            return_type = signature[-1]

        return Signature(params, return_type)

    def can_be_compiled(self):
        """
        Return whether we have the argument types, so we can compile the
        function.
        """
        for name, type_ in self.py_signature.parameters:
            if type_ is inspect._empty:
                return False

        return True

    def compile(self, verbose=0, *args):
        # (1) Python AST
        self.py_source = inspect.getsource(self.py_function)
        if verbose:
            print('====== Source ======')
            print(self.py_source)

        node = ast.parse(self.py_source)

        # (2) Infer return type if not given
        return_type = self.py_signature.return_type
        if return_type is inspect._empty:
            node.return_type = return_type
            InferVisitor(verbose).traverse(node)
            return_type = node.return_type
            self.py_signature.return_type = return_type

        # (3) The IR signature
        self.ir_module = ir.Module()
        nargs = len(args)
        params = []
        for i, (name, type_) in enumerate(self.py_signature.parameters):
            # Get type from argument if not given explicitely
            arg = args[i] if i < nargs else None
            if type_ is inspect._empty:
                assert arg is not None
                type_ = types.value_to_ir_type(arg)
                self.py_signature.parameters[i] = Parameter(name, type_)

            # IR signature
            if type(type_) is type and issubclass(type_, types.ArrayType):
                dtype = type_.dtype
                dtype = types.type_to_ir_type(dtype).as_pointer()
                params.append(Parameter(name, dtype))
                for n in range(type_.ndim):
                    params.append(Parameter(f'{name}_{n}', types.int64))
            elif type(type_) is type and issubclass(type_, types.StructType):
                dtype = self.llvm.get_dtype(self.ir_module, type_)
                params.append(Parameter(name, dtype))
            elif getattr(type_, '__origin__', None) is typing.List:
                dtype = type_.__args__[0]
                dtype = types.type_to_ir_type(dtype).as_pointer()
                params.append(Parameter(name, dtype))
                params.append(Parameter(f'{name}_0', types.int64))
            else:
                dtype = types.type_to_ir_type(type_)
                params.append(Parameter(name, dtype))

        return_type = types.type_to_ir_type(return_type)
        ir_signature = Signature(params, return_type)
        self.ir_signature = ir_signature

        # (4) For libffi
        self.nargs = len(params)
        self.argtypes = [p.type for p in params]
        self.argtypes = [
            ('p' if x.is_pointer else x.intrinsic_name)
            for x in self.argtypes]

        if return_type is types.void:
            self.rtype = ''
        elif return_type.is_pointer:
            self.rtype = 'p'
        else:
            self.rtype = return_type.intrinsic_name

        # (5) Load functions
        node.compiled = {}
        ft_f64 = ir.FunctionType(types.float64, (types.float64,))
        fs = [math.acos, math.asin, math.atan, math.atan2, math.cos, math.cosh,
              math.sin, math.sinh, math.tan, math.tanh]
        for f in fs:
            name = f.__name__
            node.compiled[f] = ir.Function(self.ir_module, ft_f64, name=name)

        for plugin in plugins:
            load_functions = getattr(plugin, 'load_functions', None)
            if load_functions is not None:
                node.compiled.update(load_functions(self.ir_module))


        # (6) The IR module and function
        f_type = ir.FunctionType(
            ir_signature.return_type,
            tuple(type_ for name, type_ in ir_signature.parameters)
        )
        ir_function = ir.Function(self.ir_module, f_type, self.name)

        # (7) AST pass: structure
        node.globals = self.globals
        node.py_signature = self.py_signature
        node.ir_signature = ir_signature
        node.ir_function = ir_function

        if verbose > 1: print('====== Debug: 1st pass ======')
        BlockVisitor(verbose).traverse(node)

        # (8) AST pass: generate
        if verbose > 1: print('====== Debug: 2nd pass ======')
        GenVisitor(verbose).traverse(node)

        # (9) IR code
        if verbose:
            print('====== IR ======')
            print(self.ir)

        # Compile
        self.mod = self.llvm.compile_ir(self.ir, self.name, verbose, self.optimize)
        address = self.llvm.engine.get_function_address(self.name)
        assert address

        # (10) C function
        self.cfunction_ptr = address
        self.prepare() # prepare libffi to call the function

        # (11) Done
        self.compiled = True

    @property
    def ir(self):
        return str(self.ir_module)

    @property
    def bc(self):
        return self.mod.as_bitcode()

    def call_args(self, *args, verbose=0):
        if self.compiled is False:
            self.compile(verbose, *args)

        c_args = []
        for py_arg in args:
            c_type = self.ir_signature.parameters[len(c_args)].type
            for plugin in plugins:
                expand_argument = getattr(plugin, 'expand_argument', None)
                if expand_argument is not None:
                    arguments = expand_argument(py_arg, c_type)
                    if arguments is not None:
                        c_args.extend(arguments)

        return tuple(c_args)


    def __call__(self, *args, verbose=0):
        c_args = self.call_args(*args, verbose=verbose)

        value = self.call(c_args)
        if verbose:
            print('====== Output ======')
            print(f'args = {args}')
            print(f'ret  = {value}')

        return value

    def prepare(self):
        pass

    def call(self, c_args):
        raise NotImplementedError


class CTypesFunction(Function):

    def prepare(self):
        type2ctypes = {
            'i64': ctypes.c_int64,
            'f64': ctypes.c_double,
            'p': ctypes.c_void_p,
            '': None,
        }

        rtype = type2ctypes[self.rtype]
        argtypes = [type2ctypes[x] for x in self.argtypes]
        assert self.nargs == len(argtypes)
        func_ptr = self.cfunction_ptr

        self.cfunc = ctypes.CFUNCTYPE(rtype, *argtypes)(func_ptr)

    def call(self, c_args):
        return self.cfunc(*c_args)


class LLVM:

    def __init__(self, fclass):
        self.engine = self.create_execution_engine()
        self.fclass = fclass
        self.dtypes = {}

    def get_dtype(self, ir_module, type_):
        type_id = id(type_)
        if type_id in self.dtypes:
            return self.dtypes[type_id]

        dtype = ir_module.context.get_identified_type(type_._name_)
        dtype.set_body(*type_.get_body())
        dtype = dtype.as_pointer()
        self.dtypes[type_id] = dtype
        return dtype

    def jit(self, py_function=None, signature=None, verbose=0, optimize=False):
        f_globals = inspect.stack()[1].frame.f_globals

        if type(py_function) is FunctionType:
            function = self.fclass(self, py_function, signature, f_globals, optimize)
            if function.can_be_compiled():
                function.compile(verbose)
            return function

        # Called as a decorator
        def wrapper(py_function):
            function = self.fclass(self, py_function, signature, f_globals, optimize)
            if function.can_be_compiled():
                function.compile(verbose)
            return function

        return wrapper

    def create_execution_engine(self):
        """
        Create an ExecutionEngine suitable for JIT code generation on
        the host CPU.  The engine is reusable for an arbitrary number of
        modules.
        """
        # Create a target machine representing the host
        target = binding.Target.from_default_triple()
        self.triple = target.triple # Keep the triple for later
        # Passing cpu, freatures and opt has not proved to be faster, but do it
        # anyway, just to show it.
        cpu = binding.get_host_cpu_name()
        features = binding.get_host_cpu_features()
        target_machine = target.create_target_machine(
            cpu=cpu,
            features=features.flatten(),
            opt=3,
        )

        # And an execution engine with an empty backing module
        backing_mod = binding.parse_assembly("")
        engine = binding.create_mcjit_compiler(backing_mod, target_machine)
        return engine

    def compile_ir(self, llvm_ir, name, verbose, optimize=False):
        """
        Compile the LLVM IR string with the given engine.
        The compiled module object is returned.
        """
        engine = self.engine

        # Create a LLVM module object from the IR
        mod = binding.parse_assembly(llvm_ir)
        mod.verify()
        # Assign triple, so the IR can be saved and compiled with llc
        mod.triple = self.triple
        if verbose:
            print('====== IR (parsed) ======')
            print(mod)

        # Optimize
        if optimize:
            pmb = binding.PassManagerBuilder()
            pmb.opt_level = 2 # 0-3 (default=2)
            pmb.loop_vectorize = True

            mpm = binding.ModulePassManager()
            # Needed for automatic vectorization
            triple = binding.get_process_triple()
            target = binding.Target.from_triple(triple)
            tm = target.create_target_machine()
            tm.add_analysis_passes(mpm)

            pmb.populate(mpm)
            mpm.run(mod)

        # Now add the module and make sure it is ready for execution
        engine.add_module(mod)
        engine.finalize_object()
        engine.run_static_constructors()

        if verbose:
            print('====== IR (optimized) ======')
            print(mod)

        return mod


# All these initializations are required for code generation!
binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()

binding.load_library_permanently("libsvml.so")
binding.set_option('', '-vector-library=SVML')


llvm = LLVM(CTypesFunction)

# Plugins (entry points)
import pkg_resources

plugins = []
for ep in pkg_resources.iter_entry_points(group='py2llvm_plugins'):
    plugin = ep.load()
    plugins.append(plugin)
