"""
Useful links:

- https://docs.python.org/3.6/library/ast.html#abstract-grammar
- http://llvmlite.pydata.org/
- https://mapping-high-level-constructs-to-llvm-ir.readthedocs.io/
- https://llvm.org/docs/LangRef.html
"""

# Standard Library
import argparse
import ast
import operator

from llvmlite import ir
from run import run


double = ir.DoubleType()
int64 = ir.IntType(64)

types = {
    float: double,
    int: int64,
}


class BaseNodeVisitor:
    """
    The ast.NodeVisitor class traverses the AST and calls user defined
    callbacks when entering a node.

    Here we do the same thing but:

    - Callback as well when exiting the node
    - Callback as well after traversing an attribute
    - Pass the parent node to the callback
    - Pass the value of the attribute to the attribute callback
    - Pass the values of all the attributes to the exit callback

    Override this class and define the callbacks you need:

    - def <classname>_enter(node, parent)
    - def <classname>_<attribute>(node, parent, value)
    - def <classname>_exit(node, parent, *args)
    - def default_enter(node, parent)
    - def default_<attribute>(node, parent, value)
    - def default_exit(node, parent, *args)

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
        method = f'{node.__class__.__name__}_{event}'
        cb = getattr(self, method, None)
        if cb is None:
            method = f'default_{event}'
            cb = getattr(self, method, None)

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
            else:
                attrs = ' '.join([repr(x) for x in args])
                print(self.depth * ' '  + f'.{event}={attrs}')

        # Call
        return cb(node, parent, *args) if cb is not None else None


class NodeVisitor(BaseNodeVisitor):

    module = None
    args = None
    builder = None
    rtype = None # Type of the return value
    local_ns = None
    ltype = None # Type of the local variable

    def print(self, line):
        print(self.depth * ' ' + line)

    def debug(self, node, parent):
        for name, field in ast.iter_fields(node):
            self.print(f'- {name} {field}')

    def Module_enter(self, node, parent):
        """
        Module(stmt* body)
        """
        assert parent is None
        self.module = ir.Module()

    def FunctionDef_enter(self, node, parent):
        """
        FunctionDef(identifier name, arguments args,
                    stmt* body, expr* decorator_list, expr? returns)
        """
        assert type(parent) is ast.Module

        # Decorators not supported
        assert not node.decorator_list

        # Return type
        # We don't support type inference yet, the return type must be
        # explicitely given with a function annotation
        assert node.returns is not None, 'return type inference is not supported'

        # Initialize function context
        self.builder = ir.IRBuilder()
        self.local_ns = {
            'float': float,
            'int': int,
        }

    def arguments_enter(self, node, parent):
        """
        arguments = (arg* args, arg? vararg, arg* kwonlyargs, expr* kw_defaults,
                     arg? kwarg, expr* defaults)
        """
        if any([node.args, node.vararg, node.kwonlyargs, node.kw_defaults, node.kwarg, node.defaults]):
            raise NotImplementedError('functions with arguments not supported')

        self.args = ()

    def FunctionDef_returns(self, node, parent, value):
        self.rtype = value
        # TODO Cache function types, do not generate twice the same
        ftype = ir.FunctionType(types[self.rtype], self.args)
        function = ir.Function(self.module, ftype, node.name)
        block = function.append_basic_block()
        self.builder = ir.IRBuilder(block)

    def Num_exit(self, node, parent, value):
        """
        Num(object n)
        """
        return value

    def Name_exit(self, node, parent, *args):
        """
        Name(identifier id, expr_context ctx)
        """
        name = node.id
        if type(node.ctx) is ast.Load:
            return self.local_ns[name]
        if type(node.ctx) is ast.Store:
            return name

    def Add_exit(self, node, parent, *args):
        return node

    def Sub_exit(self, node, parent, *args):
        return node

    def Mult_exit(self, node, parent, *args):
        return node

    def Div_exit(self, node, parent, *args):
        return node

    def BinOp_exit(self, node, parent, left, op, right):
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

#       d = {
#           (ast.Add,  int  ): self.builder.add,
#           (ast.Sub,  int  ): self.builder.sub,
#           (ast.Mult, int  ): self.builder.mul,
#           (ast.Div,  int  ): self.builder.sdiv,
#           (ast.Add,  float): self.builder.fadd,
#           (ast.Sub,  float): self.builder.fsub,
#           (ast.Mult, float): self.builder.fmul,
#           (ast.Div,  float): self.builder.fdiv,
#       }
#       f = d.get((op.__class__, self.ltype))

#       if f is None:
#           raise NotImplementedError(
#               f'{op.__class__.__name__} operator for {self.ltype} type not implemented')

#       return f(left, right)

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
        #assert type(value) is ir.values.Constant
        # LLVM does not support simple assignment to local variables.
        self.local_ns[name] = value

    def Return_enter(self, node, parent):
        self.ltype = self.rtype

    def Return_exit(self, node, parent, value):
        """
        Return(expr? value)
        """
        value = self.rtype(value)
        value = self.__py2ir(value)

        self.ltype = None
        return self.builder.ret(value)

    def __py2ir(self, value):
        py_type = type(value)
        #py_type = self.ltype
        ir_type = types[py_type]
        return ir.Constant(ir_type, value)


def node_to_llvm(node, debug=True):
    visitor = NodeVisitor(debug)
    visitor.traverse(node)
    return visitor.module

def py2llvm(node, debug=True):
    node = ast.parse(node)
    return node_to_llvm(node, debug=debug)


source = """
def f() -> int:
    a: int = 2
    b: int = 4
    c: int = (a + b) * 2 - 3
    c: int = c / 3
    return c
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    debug = args.debug

    print('====== Source ======')
    print(source)
    if debug:
        print('====== Debug ======')
    module = py2llvm(source, debug=debug)
    module = str(module)
    print('====== IR ======')
    print(module)
    print('====== Output ======')
    print(run(module, 'f'))
