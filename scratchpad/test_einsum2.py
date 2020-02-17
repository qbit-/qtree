import numpy as np
from einsum2 import einsum2
from threading import Thread
import psutil
from time import time, sleep
from profilers import timing, cpu_util, proc_count, repeat


@proc_count()
@cpu_util()
def do_einsum(n_dim=4, dim_size=2, einsum=np.einsum):
    shape = [dim_size]*(n_dim + 1)
    idx1 = [0] + list(range(1, n_dim + 1))
    idx2 = [0] + list(range(n_dim+2, 2*(n_dim+1)))
    result_idx = list(set(idx1 + idx2))
    T1 = np.random.randn(*shape)
    T2 = np.random.randn(*shape)
    print(shape)

    res = einsum(T1, idx1, T2, idx2, result_idx)
    print('res shape', res.shape)
    return res

@proc_count()
@cpu_util()
def do_einsum_flat(n_dim=4, dim_size=2, einsum=np.einsum):
    shape = [dim_size] + [dim_size**n_dim]
    idx1 = [0] + [1]
    idx2 = [0] + [2]
    result_idx = list(set(idx1 + idx2))
    T1 = np.random.randn(*shape)
    T2 = np.random.randn(*shape)
    print(shape)

    res = einsum(T1, idx1, T2, idx2, result_idx)
    print('res shape', res.shape)
    return res

def profile_einsum(einsum, ndims=range(10, 16), dim_size=2):
    msg_templ = '{ndim} dimensions of size {dim_size} total len: {totlen}'
    for ndim in ndims:
        msg = msg_templ.format(  ndim=ndim
                               , dim_size=dim_size
                               , totlen=dim_size**ndim
                              )
        with timing('Flat: '+msg):
            do_einsum_flat(n_dim=ndim, dim_size=dim_size, einsum=einsum)
        with timing(msg):
            do_einsum(n_dim=ndim, dim_size=dim_size, einsum=einsum)
        print('\n')

def profile_both():
    print('numpy')
    profile_einsum(np.einsum)
    print('einsum2')
    profile_einsum(einsum2)

if __name__=='__main__':
    profile_both()
