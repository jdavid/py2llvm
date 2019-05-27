import _testa


def run(f, *args):
    args = f.call_args(*args)
    return _testa.run(f, args)
