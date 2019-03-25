"""
This program translates a small subset of Python to LLVM IR.

Notes:

- Only a very small subset is implemented.
- Only integers and floats are supported.
- Integers are translated to 64 bit integers in LLVM, and floats are translated
  to 64 bit doubles in LLVM.
- Function arguments and return values *must* be typed (type hints).

Literals:

- The "2" literal is an integer, the "2.0" literal is a float.
- No other literal is allowed (None, etc.)

Types:

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
import argparse
import ast
import ctypes
import operator

from llvmlite import ir
from run import run


double = ir.DoubleType()
int64 = ir.IntType(64)
zero_int64 = ir.Constant(int64, 0)
zero_double = ir.Constant(double, 0)

type_py2ir = {
    float: double,
    int: int64,
}

type_ir2py = {
    double: float,
    int64: int,
}

type_py2c = {
    float: ctypes.c_double,
    int: ctypes.c_int64,
}


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
        self.callback('enter', node, parent)
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

        # Debug
        if self.debug:
            name = node.__class__.__name__
            if event == 'enter':
                if node._fields:
                    attrs = ' '.join(f'{k}' for k, v in ast.iter_fields(node))
                    print(self.depth * ' '  + f'<{name} {attrs}>')
                else:
                    print(self.depth * ' '  + f'<{name}>')
            elif event == 'exit':
                print(self.depth * ' '  + f'</{name}>')
#               if args:
#                   attrs = ' '.join(repr(x) for x in args)
#                   print(self.depth * ' '  + f'</{name} {attrs}>')
#               else:
#                   print(self.depth * ' '  + f'</{name}>')
            elif event == 'visit':
                if node._fields:
                    attrs = ' '.join(f'{k}' for k, v in ast.iter_fields(node))
                    print(self.depth * ' '  + f'<{name} {attrs} />')
                else:
                    print(self.depth * ' '  + f'<{name} />')
            else:
                if cb is not None:
                    attrs = ' '.join([repr(x) for x in args])
                    print(self.depth * ' '  + f'.{event}={attrs}')

        # Call
        return cb(node, parent, *args) if cb is not None else None


class BaseVisitor(BaseNodeVisitor):

    def FunctionDef_enter(self, node, parent):

        """
        FunctionDef(identifier name, arguments args,
                    stmt* body, expr* decorator_list, expr? returns)
        """
        assert type(parent) is ast.Module, 'nested functions not implemented'
        assert not node.decorator_list, 'decorators not implemented'
        assert node.returns is not None, 'expected type-hint for function return type'

        # Initialize function context
        node.local_ns = {'float': float, 'int': int}
        self.local_ns = node.local_ns


class BlockVisitor(BaseVisitor):
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
            return self.local_ns.get(name)

        return None

    def Module_enter(self, node, parent):
        """
        Module(stmt* body)
        """
        self.root = node

        # This will be the final output of the whole process
        node.module = ir.Module()
        node.c_functypes = {} # Signatures for ctypes


    def arguments_enter(self, node, parent):
        """
        arguments = (arg* args, arg? vararg, arg* kwonlyargs, expr* kw_defaults,
                     arg? kwarg, expr* defaults)
        """
        if any([node.vararg, node.kwonlyargs, node.kw_defaults, node.kwarg, node.defaults]):
            raise NotImplementedError('only positional arguments are supported')

    def arg_enter(self, node, parent):
        """
        arg = (identifier arg, expr? annotation)
               attributes (int lineno, int col_offset)
        """
        assert node.annotation is not None, 'function arguments must have a type hint'

    def arg_exit(self, node, parent, arg, annotation):
        return arg, annotation

    def arguments_exit(self, node, parent, args, vararg, kwonlyargs, kw_defaults, kwarg, defaults):
        return args

    def FunctionDef_args(self, node, parent, args):
        self.f_args = args

    def FunctionDef_returns(self, node, parent, returns):
        """
        When we reach this point we have all the function signature: arguments
        and return type.
        """
        root = self.root
        f_name = node.name
        f_args = self.f_args

        # Create the function type
        args = tuple(type_py2ir[py_type] for name, py_type in f_args)
        function_type = ir.FunctionType(type_py2ir[returns], args)

        # Create the function
        function = ir.Function(root.module, function_type, f_name)

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
            node.local_ns[name] = ptr

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
        node.f_rtype = returns

        # Keep the ctypes signature, used in the tests
        c_args = [returns] + [py_type for name, py_type in f_args]
        c_args = [type_py2c[t] for t in c_args]
        root.c_functypes[f_name] = ctypes.CFUNCTYPE(*c_args)

    def If_test(self, node, parent, test):
        """
        If(expr test, stmt* body, stmt* orelse)
        """
        block = self.function.append_basic_block()
        node.block_true = block

    def If_body(self, node, parent, body):
        block = self.function.append_basic_block()
        node.block_false = block

    def If_orelse(self, node, parent, orelse):
        block = self.function.append_basic_block()
        node.block_next = block


class GenVisitor(BaseVisitor):
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

    def __get_type(self, value):
        """
        Return the type of value, only int and float are supported.
        The value may be a Python or an IR value.
        """
        if isinstance(value, ir.Value):
            vtype = {
                ir.IntType: int,
                ir.DoubleType: float
            }.get(value.type.__class__)
        else:
            vtype = type(value)

        if vtype in (int, float):
            return vtype

        raise NotImplementedError(f'only int and float supported, got {repr(value)}')

    def __get_lr_type(self, left, right):
        """
        For coercion purposes, return int if both are int, float if one is
        float.
        """
        ltype = self.__get_type(left)
        rtype = self.__get_type(right)
        if ltype is int and rtype is int:
            return int
        return float

    def __py2ir(self, value, dst_type=None):
        src_type = self.__get_type(value)

        # If target type is not given, will be same as source type
        if dst_type is None:
            dst_type = src_type

        # Target type, in IR
        dst_type_ir = type_py2ir[dst_type]

        # If Python value, return a constant
        if not isinstance(value, ir.Value):
            if src_type is not dst_type:
                value = dst_type(value)

            return ir.Constant(dst_type_ir, value)

        # Coerce
        if dst_type is not src_type:
            conversion = {
                (int, float): self.builder.sitofp,
                (float, int): self.builder.fptosi,
            }.get((src_type, dst_type))
            if conversion is None:
                err = f'Conversion from {src_type} to {dst_type} not suppoerted'
                raise NotImplementedError(err)

            value = conversion(value, dst_type_ir)

        return value

    def noop(self):
        """
        This is to support empty blocks, e.g. "if cond: pass"
        """
        self.builder.add(zero_int64, zero_int64)

    def assign(self, name, value):
        """
        This is to support simple assignments, e.g. "b = a"
        """
        py_type = self.__get_type(value)
        if py_type is int:
            var = self.builder.add(value, zero_int64, name=name)
        elif py_type is float:
            var = self.builder.fadd(value, zero_double, name=name)

        var.name = name
        return var

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
            value = self.local_ns[name]
            if type(value) is ir.AllocaInstr:
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

    def operator_visit(self, node, parent):
        return node

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

    def object_visit(self, node, parent):
        raise NotImplementedError(f'{node.__class__} not supported')

    #
    # Expressions
    #

    def FunctionDef_enter(self, node, parent):
        self.local_ns = node.local_ns
        self.builder = node.builder
        self.f_rtype = node.f_rtype
        self.block_vars = node.block_vars

    def FunctionDef_exit(self, node, parent, *args):
        node.builder.position_at_end(node.block_vars)
        node.builder.branch(node.block_start)

    def BinOp_exit(self, node, parent, left, op, right):
        # Two Python values
        if not isinstance(left, ir.Value) and not isinstance(right, ir.Value):
            ast2op = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
            }
            py_op = ast2op.get(op.__class__)
            if py_op is None:
                raise NotImplementedError(
                    f'{op.__class__.__name__} operator for {self.ltype} type not implemented')
            return py_op(left, right)

        # One or more IR values
        py_type = self.__get_lr_type(left, right)
        left = self.__py2ir(left, py_type)
        right = self.__py2ir(right, py_type)

        d = {
            (ast.Add,  int  ): self.builder.add,
            (ast.Sub,  int  ): self.builder.sub,
            (ast.Mult, int  ): self.builder.mul,
            (ast.Div,  int  ): self.builder.sdiv,
            (ast.Add,  float): self.builder.fadd,
            (ast.Sub,  float): self.builder.fsub,
            (ast.Mult, float): self.builder.fmul,
            (ast.Div,  float): self.builder.fdiv,
        }
        ir_op = d.get((op.__class__, py_type))
        if ir_op is None:
            raise NotImplementedError(
                f'{op.__class__.__name__} operator for {self.ltype} type not implemented')

        return ir_op(left, right)

    def Compare_exit(self, node, parent, left, ops, comparators):
        """
        Compare(expr left, cmpop* ops, expr* comparators)
        """
        assert len(ops) == 1
        assert len(comparators) == 1
        op = ops[0]
        right = comparators[0]

        # Two Python values
        if not isinstance(left, ir.Value) and not isinstance(right, ir.Value):
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

        # One or more IR values
        py_type = self.__get_lr_type(left, right)
        left = self.__py2ir(left, py_type)
        right = self.__py2ir(right, py_type)

        d = {
            int: self.builder.icmp_signed,
            float: self.builder.fcmp_unordered, # XXX fcmp_ordered
        }
        ir_op = d.get(py_type)
        return ir_op(op, left, right)


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
        #self.noop()

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
        # LLVM does not support simple assignment to local variables.
        self.local_ns[target] = value
        self.ltype = None

    def Assign_enter(self, node, parent):
        """
        Assign(expr* targets, expr value)
        """
        assert len(node.targets) == 1
        assert type(node.targets[0]) is ast.Name
        assert type(node.targets[0].ctx) is ast.Store

    def Assign_exit(self, node, parent, name, value):
        name = name[0]
        py_type = self.__get_type(value)
        ir_type = type_py2ir[py_type]

        ptr = self.local_ns.get(name)
        if ptr is None:
            block_cur = self.builder.block
            self.builder.position_at_end(self.block_vars)
            ptr = self.builder.alloca(ir_type, name=name)
            self.builder.position_at_end(block_cur)
            self.local_ns[name] = ptr

        return self.builder.store(value, ptr)

    def Return_enter(self, node, parent):
        self.ltype = self.f_rtype

    def Return_exit(self, node, parent, value):
        """
        Return(expr? value)
        """
        value = self.__py2ir(value, self.f_rtype)

        self.ltype = None
        return self.builder.ret(value)


def py2llvm(node, debug=True):
    # Source to AST tree
    node = ast.parse(node)

    # 1st pass: structure
    if debug:
        print('====== Debug: 1st pass ======')
    BlockVisitor(debug).traverse(node)

    # 2nd pass: generate
    if debug:
        print('====== Debug: 2nd pass ======')
    GenVisitor(debug).traverse(node)

    # Return IR (and ctypes signatures)
    llvm_ir = str(node.module)
    return llvm_ir, node.c_functypes


source = """
def f(a: int) -> int:
    if a == 0:
        b = a + 1
    elif a == 1:
        b = a * 2
    elif a == 2:
        return 3
    else:
        b = a

    return b

#   b: int = 4
#   c: int = (a + b) * 2 - 3
#   c: int = c / 3
#   return c
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    debug = args.debug

    print('====== Source ======')
    print(source)
    llvm_ir, sigs = py2llvm(source, debug=debug)
    print('====== IR ======')
    print(llvm_ir)
    print('====== Output ======')
    fname = 'f'
    run(llvm_ir, fname, sigs[fname], 0, debug=True)
    run(llvm_ir, fname, sigs[fname], 1, debug=True)
    run(llvm_ir, fname, sigs[fname], 2, debug=True)
    run(llvm_ir, fname, sigs[fname], 3, debug=True)
