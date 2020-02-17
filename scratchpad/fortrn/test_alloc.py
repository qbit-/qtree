import allocarr as alar
import numpy as np
import sys
sys.path.append('.')
import profilers as pr

alar.alloc.a = np.random.randn(2,2**13)
alar.alloc.b = np.random.randn(3,2**13)
alar.alloc.c = np.zeros((6,2**26))

with pr.timing('Time fortran'):
    c = alar.alloc.foo()
print(c, alar.alloc.c)
with pr.timing('time np'):
    c  =np.kron(alar.alloc.a,alar.alloc.b)
print(np.max(c-alar.alloc.c))
