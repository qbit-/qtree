from multiprocessing import Pool, Array
import numpy as np
from profilers import timing, cpu_util, proc_count, repeat, timed
import os

### Utils

def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj())

def multiply_list(l):
    r = 1
    for x in l:
        r *= x
    return r

def full_slice():
    return slice(*[None]*3)

def multidim_slice(starts, ends, steps=None):
    assert len(starts)==len(ends)
    dim_cnt = len(starts)
    if steps == None:
        steps = [1]*dim_cnt
    s = ( slice(*x) for x in zip(starts, ends, steps) )
    return s

def slices_at(I):
    return (slice(s,e) for s,e in zip(I[:-1], I[1:]))

def multidim_chunks(shape, N):
    """ returns a list of slices for a multidim array
    """
    flat = multiply_list(shape)
    flat_ends = np.linspace(flat, N+1)
    mdim_ends = np.unravel_index(flat_ends, shape)
    # get tuple of dimensional arrays of slices
    mdim_slices = ( slices_at(dim) for dim in mdim_ends)
    # return array of tuples slices with l[i] = (dim1, dim2, ...)
    return zip(*mdim_slices)

###
### Workers

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
    with timing('assign'):
        os.global_C[target_slice] = x

def unpacked(args):
    return contract_put(*args)

### 

def eliminate(A, B, nproc=3):
    #ops = [A,B]
    N = A.shape[0]
    target_shape = [N]
    target_shape += list(A.shape[1:])
    target_shape += list(B.shape[1:])
    print('target shape', target_shape)
    # Create shared memory array
    with timing('Create array'):
        flat_size = multiply_list(target_shape)
        os.global_C = tonumpyarray(Array('d',flat_size))
    os.global_C = os.global_C.reshape((N, multiply_list(A.shape[1:]), multiply_list(B.shape[1:])))

    ops = [A, B]
    ops = [x.reshape(N, -1) for x in ops]

    #chunked = [chunk_slice(x, nproc) for x in ops]
    chunk_a = [ops[0] for _ in range(nproc)]
    chunk_b, slices_b= chunk_slice(ops[1], nproc)
    target_slices = [(full_slice(), full_slice(), sl) for sl in slices_b]
    chunked = [chunk_a, chunk_b]

    args = zip(*chunked, target_slices)
    pool = Pool(processes=nproc)
    with timing('Parallel work'):
        _ = pool.map(unpacked, args)
    print('Done\n')
    return os.global_C.reshape(target_shape)


def chunk_slice(X, nproc):
    _shape = X.shape[1:]
    _len = sum(_shape)
    _splits = np.linspace(0, _len, nproc+1, dtype=int)
    print('Splits:', _splits)
    slices = [slice(s,e) for s,e in  zip(_splits[:-1], _splits[1:])]

    _splitted = [X[:,slic] for slic in slices]
    return _splitted, slices

def test_2dim():
    A = np.random.randn(4, int(10e1))
    B = np.random.randn(4, 4000000)
    with timing('Parallel total'):
        C = eliminate(A,B)
    print('result shape', C.shape)
    with timing('Simple'):
        C1 = first_idx_contract(A,B)
    assert np.array_equal(C , C1)

def test_3dim():
    A = np.random.randn(4, 700, 10)
    B = np.random.randn(4, 600, 10)
    with timing('Parallel total'):
        C = eliminate(A,B)
    print('result shape', C.shape)
    with timing('Simple'):
        C1 = np.einsum('abc, aef -> abcef',A, B)
    assert np.array_equal(C , C1)

if __name__=='__main__':
    test_3dim()
