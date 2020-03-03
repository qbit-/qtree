from multiprocessing import Pool
import numpy as np
from profilers import timing, cpu_util, proc_count, repeat, timed
import profilers
import os


@timed('contract')
@cpu_util()
def first_idx_contract(A,B):
    assert A.shape[0] == B.shape[0]
    pid = os.getpid()
    print(pid,'shape',A.shape,B.shape)
    x =  np.einsum('ij, ik -> ijk', A,B)
    print('cntr')
    return x

def unpacked(args):
    return first_idx_contract(*args)

def eliminate(A, B, nproc=16):
    #ops = [A,B]
    N = A.shape[0]
    target_shape = [N]
    target_shape += list(A.shape[1:])
    target_shape += list(B.shape[1:])
    print('target shape', target_shape)
    A, B = A.reshape((N,-1)), B.reshape((N,-1))

    ops = [A, B]
    #chunked = [chunk_slice(x, nproc) for x in ops]
    chunk_a = [A for _ in range(nproc)]
    chunk_b = chunk_slice(B, nproc)
    chunked = [chunk_a, chunk_b]

    pool = Pool(processes=nproc)
    args = zip(*chunked)
    result = pool.map(unpacked, args)
    print('Done\n')
    return np.concatenate(result, axis=-1)


def chunk_slice(X, nproc):
    _shape = X.shape[1:]
    _len = sum(_shape)
    _splits = np.linspace(0, _len, nproc+1, dtype=int)
    print('Splits:', _splits)
    _splitted = [X[:,s:e] for s, e in zip(_splits[:-1], _splits[1:])]
    return _splitted

def test_par():
    A = np.random.randn(4, int(10e1))
    B = np.random.randn(4, 1000000)
    with timing('Parallel'):
        C = eliminate(A,B)
    print('result shape', C.shape)
    with timing('Simple'):
        C1 = first_idx_contract(A,B)
    assert np.array_equal(C , C1)

if __name__=='__main__':
    test_par()
