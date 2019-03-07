import ast

from llvmlite import ir


double = ir.DoubleType()
int64 = ir.IntType(64)

types = {
    'float': double,
    'int': int64,
}

def print_fields(node):
    for field in ast.iter_fields(node):
        print(f'- {field}')


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
        # Enter
        self.enter(node, parent)
        # Traverse
        value = None
        for field, value in self.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        value = self.traverse(item, node)
            elif isinstance(value, ast.AST):
                value = self.traverse(value, node)
        # Exit
        return self.exit(node, parent, value)

    def enter(self, node, parent):
        method = 'enter_' + node.__class__.__name__
        visitor = getattr(self, method, self.enter_generic)
        return visitor(node, parent)

    def exit(self, node, parent, value):
        method = 'exit_' + node.__class__.__name__
        visitor = getattr(self, method, self.exit_generic)
        return visitor(node, parent, value)

    def enter_generic(self, node, parent):
        """Called if no explicit enter function exists for a node."""

    def exit_generic(self, node, parent, value):
        """Called if no explicit exit function exists for a node."""


class NodeVisitor(BaseNodeVisitor):

    module = None
    args = None
    builder = None
    rtype = None

    def enter_generic(self, node, parent):
        print(f'ENTER {node}')
        #print(node.__class__.__mro__)
        #print_fields(node)
        #print()

    def exit_generic(self, node, parent, value):
        print(f'EXIT  {node}')

    def enter_Module(self, node, parent):
        """
        Module(stmt* body)
        """
        assert parent is None
        self.enter_generic(node, parent)
        self.module = ir.Module()

    def enter_FunctionDef(self, node, parent):
        """
        FunctionDef(identifier name, arguments args,
                    stmt* body, expr* decorator_list, expr? returns)
        """
        assert type(parent) is ast.Module
        self.enter_generic(node, parent)

        # Decorators not supported
        assert not node.decorator_list

        # Return type
        # We don't support type inference yet, the return type must be
        # explicitely given with a function annotation
        assert node.returns is not None, 'return type inference is not supported'
        assert type(node.returns) is ast.Name, 'only name suported for now'
        assert type(node.returns.ctx) is ast.Load
        self.rtype = types[node.returns.id]

        self.builder = ir.IRBuilder()

    def enter_arguments(self, node, parent):
        """
        arguments = (arg* args, arg? vararg, arg* kwonlyargs, expr* kw_defaults,
                     arg? kwarg, expr* defaults)
        """
        self.enter_generic(node, parent)

        if any([node.args, node.vararg, node.kwonlyargs, node.kw_defaults, node.kwarg, node.defaults]):
            raise NotImplementedError('functions with arguments not supported')

        self.args = ()

    def exit_arguments(self, node, parent, value):
        self.exit_generic(node, parent, value)
        # TODO Cache function types, do not generate twice the same
        ftype = ir.FunctionType(self.rtype, self.args)
        function = ir.Function(self.module, ftype, parent.name)
        block = function.append_basic_block()
        self.builder = ir.IRBuilder(block)

    def exit_Return(self, node, parent, value):
        """
        Return(expr? value)
        """
        self.exit_generic(node, parent, value)
        self.builder.ret(value)

    def enter_Name(self, node, parent):
        """
        Name(identifier id, expr_context ctx)
        """
        assert type(node.ctx) is ast.Load
        self.enter_generic(node, parent)
        print('**', node.id)

    def exit_Num(self, node, parent, value):
        """
        Num(object n)
        """
        return ir.Constant(double, node.n) # TODO Type inference


def py2llvm(source):
    node = ast.parse(source)
    visitor = NodeVisitor()
    visitor.traverse(node)
    print(visitor.module)



simple = """
def f() -> float:
    return 42
"""

if __name__ == '__main__':
    py2llvm(simple)
