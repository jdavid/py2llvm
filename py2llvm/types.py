from llvmlite import ir

try:
    import numpy as np
except ImportError:
    np = None


# Basic IR types
void = ir.VoidType()
float32 = ir.FloatType()
float64 = ir.DoubleType()
int32 = ir.IntType(32)
int64 = ir.IntType(64)


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
