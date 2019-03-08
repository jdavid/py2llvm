"""
Useful links:

- https://docs.python.org/3.6/library/ast.html#abstract-grammar
- http://llvmlite.pydata.org/
- https://mapping-high-level-constructs-to-llvm-ir.readthedocs.io/
- https://llvm.org/docs/LangRef.html
"""

import ast
from llvmlite import ir
from run import run


double = ir.DoubleType()
int64 = ir.IntType(64)

types = {
    'float': double,
    'int': int64,
}


class BaseNodeVisitor:
    """
    The ast.NodeVisitor class is not very useful because:

    - It doesn't keep context
    - It visits the node only once

    Here we fix these problems.

    Override this class and define the following as you need:

    - def enter_<classname>(node, parent)
    - def exit_<classname>(node, parent, value)
    - def enter_generic(node, parent)
    - def exit_generic(node, parent, value)

    Call using traverse:

        class NodeVisitor(BaseNodeVisitor):
            ...

        node = ast.parse(source)
        NodeVisitor().traverse(node)
    """

    def __init__(self):
        self.depth = 0

    @classmethod
    def get_fields(cls, node):
        fields = {
            # Do not traverse returns
            ast.FunctionDef: ('args', 'body', 'decorator_list'),
            # Traverse returns before body
            #ast.FunctionDef: ('args', 'returns', 'body', 'decorator_list'),
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
        # Debug
        if node._fields:
            fields = ' '.join(f'{k}=' for k, v in ast.iter_fields(node))
            print(self.depth * ' '  + f'<{node.__class__.__name__} {fields}>')
        else:
            print(self.depth * ' '  + f'<{node.__class__.__name__}>')

        self.depth += 1

        # Enter
        self.enter(node, parent)

        # Traverse
        args = []
        for name, field in self.iter_fields(node):
            if isinstance(field, list):
                tmp = [self.traverse(x, node) for x in field if isinstance(x, ast.AST)]
                args.append(tmp)
            elif isinstance(field, ast.AST):
                tmp = self.traverse(field, node)
                args.append(tmp)

        # Debug
        self.depth -= 1
        if args:
            tmp = ' '.join(repr(x) for x in args)
            print(self.depth * ' '  + f'</{node.__class__.__name__} {tmp}>')
        else:
            print(self.depth * ' '  + f'</{node.__class__.__name__}>')

        # Exit
        return self.exit(node, parent, *args)

    def enter(self, node, parent):
        method = 'enter_' + node.__class__.__name__
        visitor = getattr(self, method, self.enter_generic)
        return visitor(node, parent)

    def exit(self, node, parent, *args):
        method = 'exit_' + node.__class__.__name__
        visitor = getattr(self, method, self.exit_generic)
        return visitor(node, parent, *args)

    def enter_generic(self, node, parent):
        """Called if no explicit enter function exists for a node."""

    def exit_generic(self, node, parent, *args):
        """Called if no explicit exit function exists for a node."""


class NodeVisitor(BaseNodeVisitor):

    module = None
    args = None
    builder = None
    rtype = None
    local_ns = None

    def print(self, line):
        print(self.depth * ' ' + line)

    def debug(self, node, parent):
        for name, field in ast.iter_fields(node):
            self.print(f'- {name} {field}')

    def enter_Module(self, node, parent):
        """
        Module(stmt* body)
        """
        assert parent is None
        self.module = ir.Module()

    def enter_FunctionDef(self, node, parent):
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
        assert type(node.returns) is ast.Name, 'only name suported for now'
        assert type(node.returns.ctx) is ast.Load, f'unexpected Name.ctx={node.returns.ctx}'

        # Initialize function context
        self.rtype = types[node.returns.id]
        self.builder = ir.IRBuilder()
        self.local_ns = {}

    def enter_arguments(self, node, parent):
        """
        arguments = (arg* args, arg? vararg, arg* kwonlyargs, expr* kw_defaults,
                     arg? kwarg, expr* defaults)
        """
        if any([node.args, node.vararg, node.kwonlyargs, node.kw_defaults, node.kwarg, node.defaults]):
            raise NotImplementedError('functions with arguments not supported')

        self.args = ()

    def exit_arguments(self, node, parent, *args):
        # TODO Cache function types, do not generate twice the same
        ftype = ir.FunctionType(self.rtype, self.args)
        function = ir.Function(self.module, ftype, parent.name)
        block = function.append_basic_block()
        self.builder = ir.IRBuilder(block)

    def exit_Num(self, node, parent, *args):
        """
        Num(object n)
        """
        return ir.Constant(double, node.n) # TODO Type inference

    def exit_Name(self, node, parent, *args):
        """
        Name(identifier id, expr_context ctx)
        """
        name = node.id
        if type(node.ctx) is ast.Load:
            return self.local_ns[name]
        if type(node.ctx) is ast.Store:
            return name

    def exit_Add(self, node, parent, *args):
        return node

    def exit_Sub(self, node, parent, *args):
        return node

    def exit_Mult(self, node, parent, *args):
        return node

    def exit_Div(self, node, parent, *args):
        return node

    def exit_BinOp(self, node, parent, left, op, right):
        f = {
            ast.Add: self.builder.fadd,
            ast.Sub: self.builder.fsub,
            ast.Mult: self.builder.fmul,
            ast.Div: self.builder.fdiv,
        }.get(type(op))

        if f is None:
            raise NotImplementedError(f'{node.op} operator not implemented')

        return f(left, right)

    def enter_Assign(self, node, parent):
        """
        Assign(expr* targets, expr value)
        """
        assert len(node.targets) == 1
        assert type(node.targets[0]) is ast.Name
        assert type(node.targets[0].ctx) is ast.Store

    def exit_Assign(self, node, parent, name, value):
        name = name[0]
        #assert type(value) is ir.values.Constant
        # LLVM does not support simple assignment to local variables.
        self.local_ns[name] = value

    def exit_Return(self, node, parent, value):
        """
        Return(expr? value)
        """
        return self.builder.ret(value)


def node_to_llvm(node):
    visitor = NodeVisitor()
    visitor.traverse(node)
    return visitor.module

def py2llvm(node):
    node = ast.parse(node)
    return node_to_llvm(node)


source = """
def f() -> float:
    a = 2
    b = 4
    c = (a + b) * 2 - 3
    c = c / 3
    return c
"""

if __name__ == '__main__':
    print('====== Source ======')
    print(source)
    print('====== Debug ======')
    module = py2llvm(source)
    module = str(module)
    print('====== IR ======')
    print(module)
    print('====== Output ======')
    print(run(module, 'f'))
