import sympy as sym
from sympy.integrals.transforms import _fourier_transform

def fourier_transform(f, x, k):
    return _fourier_transform(f, x, k, 1, -1, None)

def inverse_fourier_transform(f, k, x):
    return _fourier_transform(f, k, -x, 1/(2*sym.pi), 1, None)