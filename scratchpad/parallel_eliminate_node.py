from multiprocessing import Pool, Array
import numpy as np
from profilers import timing, cpu_util, proc_count, repeat, timed
import profilers
import os

from shared_ndarray import SharedNDArray

def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj())

@timed('contract')
@cpu_util()
def first_idx_contract(A,B):
    assert A.shape[0] == B.shape[0]
    pid = os.getpid()
    print(pid,'shape',A.shape,B.shape)
    x =  np.einsum('ij, ik -> ijk', A,B)
    return x

def contract_put(A, B, target_slice):
    x = first_idx_contract(A,B)
    os.global_C[target_slice] = x

def unpacked(args):
    return contract_put(*args)

def eliminate(A, B, nproc=32):
    #ops = [A,B]
    N = A.shape[0]
    target_shape = [N]
    target_shape += list(A.shape[1:])
    target_shape += list(B.shape[1:])
    print('target shape', target_shape)
    with timing('Create array'):
        os.global_C = tonumpyarray(Array('d', N*target_shape[1]*target_shape[2]))
    os.global_C = os.global_C.reshape(target_shape)

    ops = [A, B]
    ops = [x.reshape(N, -1) for x in ops]

    #chunked = [chunk_slice(x, nproc) for x in ops]
    chunk_a = [A for _ in range(nproc)]
    chunk_b, slices_b= chunk_slice(B, nproc)
    target_slices = [(slice(None,None,None), slice(None,None,None), sl) for sl in slices_b]
    chunked = [chunk_a, chunk_b]

    args = zip(*chunked, target_slices)
    pool = Pool(processes=nproc)
    with timing('Parallel work'):
        result = pool.map(unpacked, args)
    print('Done\n')
    return os.global_C


def chunk_slice(X, nproc):
    _shape = X.shape[1:]
    _len = sum(_shape)
    _splits = np.linspace(0, _len, nproc+1, dtype=int)
    print('Splits:', _splits)
    slices = [slice(s,e) for s,e in  zip(_splits[:-1], _splits[1:])]

    _splitted = [X[:,slic] for slic in slices]
    return _splitted, slices

def test_par():
    A = np.random.randn(4, int(10e1))
    B = np.random.randn(4, 4000000)
    with timing('Parallel total'):
        C = eliminate(A,B)
    print('result shape', C.shape)
    with timing('Simple'):
        C1 = first_idx_contract(A,B)
    assert np.array_equal(C , C1)

if __name__=='__main__':
    test_par()
