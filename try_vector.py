"""
Usage:

  python try_vector.py -v
"""

import argparse
import numpy as np
import py2llvm as llvm


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='count', default=0)
args = parser.parse_args()
verbose = args.verbose

print(verbose)

#@llvm.jit(verbose=verbose)
#def f(a1, a2, out):
#    for i in range(a1.shape[0]):
#        if a1[i] > a2[i]:
#            out[i] = a1[i] + a2[i] * 2
#        else:
#            out[i] = a1[i] - a2[i]
#
#    return 1

@llvm.jit(verbose=verbose, optimize=True)
def f(a1, a2, out):
    for i in range(4000):
        out[i] = a1[i] + a2[i]

    return 1


# Prepare function arguments
a1 = [1.0, 2.5, 1.3, 5.2] * 1000
a2 = [0.5, 1.5, 2.2, 2.9] * 1000
a1 = np.array(a1, dtype=np.float64)
a2 = np.array(a2, dtype=np.float64)
out = np.empty((4000,), dtype=np.float64)

f(a1, a2, out, verbose=verbose) # COMPILED
print(a1)
print(a2)
print(out)
