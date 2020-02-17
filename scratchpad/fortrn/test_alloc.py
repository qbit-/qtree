import allocarr as alar
import numpy as np
from einsum2 import einsum2
import sys
sys.path.append('.')
import profilers as pr

rank = int(sys.argv[1])

alar.alloc.a = np.random.randn(2,2**rank)
alar.alloc.b = np.random.randn(3,2**rank)
alar.alloc.c = np.zeros((6,2**(rank*2)))

with pr.timing('Time fortran'):
    c = alar.alloc.foo()
print(c, alar.alloc.c)

with pr.timing('time np'):
    c  =np.kron(alar.alloc.a,alar.alloc.b)

with pr.timing('time einsum'):
    c  =np.einsum('ij,kl->ijkl', alar.alloc.a,alar.alloc.b)

with pr.timing('time einsum2'):
    c  = einsum2('ij,kl->ijkl', alar.alloc.a,alar.alloc.b)
