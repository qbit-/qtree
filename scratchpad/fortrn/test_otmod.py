from OTfor import otmod
import numpy as np
from einsum2 import einsum2
import sys
sys.path.append('.')
import profilers as pr

rank = int(sys.argv[1])

x = np.empty((2,2**rank))
y = np.empty((3,2**rank))
otmod.a = x
otmod.b = y
otmod.c = np.zeros((6,2**(rank*2)))

with pr.timing('Time fortran'):
    c = otmod.pra()
print(c, otmod.c)

with pr.timing('time np'):
    c  =np.kron(x,y)

with pr.timing('time einsum'):
    c  =np.einsum('ij,kl->ijkl',x,y)

with pr.timing('time einsum2'):
    c  = einsum2('ij,kl->ijkl', x,y)
