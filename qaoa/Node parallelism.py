# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Slice-specified-nodes-in-dimspec" data-toc-modified-id="Slice-specified-nodes-in-dimspec-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Slice specified nodes in dimspec</a></span></li><li><span><a href="#Test-parallelism" data-toc-modified-id="Test-parallelism-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Test parallelism</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Example-task" data-toc-modified-id="Example-task-2.0.1"><span class="toc-item-num">2.0.1&nbsp;&nbsp;</span>Example task</a></span></li></ul></li><li><span><a href="#Serial-invocation" data-toc-modified-id="Serial-invocation-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Serial invocation</a></span><ul class="toc-item"><li><span><a href="#One-var-parallelisation" data-toc-modified-id="One-var-parallelisation-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>One var parallelisation</a></span></li><li><span><a href="#Two-var-parallelisation" data-toc-modified-id="Two-var-parallelisation-2.1.2"><span class="toc-item-num">2.1.2&nbsp;&nbsp;</span>Two var parallelisation</a></span></li><li><span><a href="#Many-var-parallelisation" data-toc-modified-id="Many-var-parallelisation-2.1.3"><span class="toc-item-num">2.1.3&nbsp;&nbsp;</span>Many var parallelisation</a></span></li></ul></li><li><span><a href="#Use-ray" data-toc-modified-id="Use-ray-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Use ray</a></span><ul class="toc-item"><li><span><a href="#Simple-assignment" data-toc-modified-id="Simple-assignment-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Simple assignment</a></span></li><li><span><a href="#Concurrent-assignment" data-toc-modified-id="Concurrent-assignment-2.2.2"><span class="toc-item-num">2.2.2&nbsp;&nbsp;</span>Concurrent assignment</a></span></li></ul></li><li><span><a href="#Use-unix-tools" data-toc-modified-id="Use-unix-tools-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Use unix tools</a></span><ul class="toc-item"><li><span><a href="#Threading" data-toc-modified-id="Threading-2.3.1"><span class="toc-item-num">2.3.1&nbsp;&nbsp;</span>Threading</a></span></li><li><span><a href="#Multiprocessing" data-toc-modified-id="Multiprocessing-2.3.2"><span class="toc-item-num">2.3.2&nbsp;&nbsp;</span>Multiprocessing</a></span></li></ul></li></ul></li></ul></div>
# -

import ray
import pyrofiler as pyrof
import numpy as np
import sys
np.random.seed(42)


# # Slice specified nodes in dimspec

# +
def _none_slice():
    return slice(None)

def _get_idx(x, idxs, slice_idx, shapes=None):
    if shapes is None:
        shapes = [2]*len(idxs)
    point = np.unravel_index(slice_idx, shapes)
    get_point = {i:p for i,p in zip(idxs, point)}
    if x in idxs:
        p = get_point[x]
        return slice(p,p+1)
    else:
        return _none_slice()

def _slices_for_idxs(idxs, *args, shapes=None, slice_idx=0):
    """Return array of slices along idxs"""
    slices = []
    for indexes in args:
        _slice = [_get_idx(x, idxs, slice_idx, shapes) for x in indexes ]
        slices.append(tuple(_slice))
    return slices
        


# +
dims1 = [1,3,4 ]
dims2 = [2,4,3, 5]
contract = [dims1, dims2]

slice_among = [4, 3]
shapes = [2, 3]

test_slices = [
    _slices_for_idxs(slice_among, *contract, shapes=shapes, slice_idx=i)
    for i in range(4)
    ]
[print(x) for x in test_slices]

# -

# # Test parallelism
# ### Example task

# +
def get_example_task():
    A = 8
    #A = 13
    B, C = 10, 7
    shape1 = [2]*(A+B)
    shape2 = [2]*(A+C)
    T1 = np.random.randn(*shape1)
    T2 = np.random.randn(*shape2)
    common = list(range(A))
    idxs1 = common + list(range(A, A+B))
    idxs2 = common + list(range(A+B, A+B+C))
    return (T1, idxs1), (T2, idxs2)

x, y = get_example_task()
x[1], y[1]


# -

# ## Serial invocation

# +

#@ray.remote
def contract(A, B):
    a, idxa = A
    b, idxb = B
    contract_idx = set(idxa) & set(idxb)
    result_idx = set(idxa + idxb)
    C = np.einsum(a,idxa, b,idxb, result_idx)
    return C

with pyrof.timing('contract'):
    C = contract(x, y)

# -

# ### One var parallelisation

# +
def sliced_contract(x, y, idxs, num):
    slices = _slices_for_idxs(idxs, x[1], y[1], slice_idx=num)
    a = x[0][slices[0]]
    b = y[0][slices[1]]
    with pyrof.timing(f'\tcontract sliced {num}'):
        C = contract((a, x[1]), (b, y[1]))
    return C


def target_slice(result_idx, idxs, num):
    slices = _slices_for_idxs(idxs, result_idx, slice_idx=num)
    return slices


# -

def _one_var_seqpar():
    contract_idx = set(x[1]) & set(y[1])
    result_idx = set(x[1] + y[1])

    with pyrof.timing(f'contract simple'):
        C = contract(x,y)

    par_vars = [1]
    target_shape = C.shape
    C0 = sliced_contract(x, y, par_vars, 0)
    C1 = sliced_contract(x, y, par_vars, 1)

    with pyrof.timing('allocate result'):
        C_par = np.empty(target_shape)
    s0 = target_slice(result_idx, par_vars, 0)
    s1 = target_slice(result_idx, par_vars, 1)

    with pyrof.timing('slice result'):
        _ = C_par[s0[0]]
        
    with pyrof.timing('assignment'):
        C_par[s0[0]] = C0
        C_par[s1[0]] = C1

    assert np.array_equal(C, C_par)


# ### Two var parallelisation

def _two_var_seqpar():
    contract_idx = set(x[1]) & set(y[1])
    result_idx = set(x[1] + y[1])

    with pyrof.timing(f'contract simple'):
        C = contract(x,y)
        
    par_vars = [1, 17]
    target_shape = C.shape
    C_patches = [
        sliced_contract(x, y, par_vars, i)
        for i in range(4)
    ]

    with pyrof.timing('allocate result'):
        C_par = np.empty(target_shape)

    patch_slces = [
        target_slice(result_idx, par_vars, i)
        for i in range(4)
    ]

    with pyrof.timing('assignment'):
        for s, patch in zip(patch_slces, C_patches):
            C_par[s[0]] = patch

    assert np.array_equal(C, C_par)


# ### Many var parallelisation

# +
contract_idx = set(x[1]) & set(y[1])
result_idx = set(x[1] + y[1])

with pyrof.timing(f'contract simple'):
    C = contract(x,y)
    
par_vars = [1, 4, 17, 5]
threads = 2**len(par_vars)
target_shape = C.shape

with pyrof.timing('Sequential patches'):
    C_patches = [
        sliced_contract(x, y, par_vars, i)
        for i in range(threads)
    ]

with pyrof.timing('allocate result'):
    C_par = np.empty(target_shape)

patch_slces = [
    target_slice(result_idx, par_vars, i)
    for i in range(threads)
]

with pyrof.timing('assignment'):
    for s, patch in zip(patch_slces, C_patches):
        C_par[s[0]] = patch

assert np.array_equal(C, C_par)

# -
# ## Use ray
# ### Simple assignment


def use_ray():
    try: 
        ray.init()
    except:
        print('ray already working')
        pass
    nodes = ray.nodes()

def ray_simple():
    contract_idx = set(x[1]) & set(y[1])
    result_idx = set(x[1] + y[1])


    with pyrof.timing(f'contract simple'):
        C = contract(x,y)
        
    sliced_contract_ray = ray.remote(sliced_contract)
        
    par_vars = [1,5,15,17]
    threads = 2**len(par_vars)
    target_shape = C.shape

    with pyrof.timing('Ray total compute'):
        with pyrof.timing('Ray compute'):
            with pyrof.timing('  Ray submit'):
                C_patches = [
                    sliced_contract_ray.remote(x, y, par_vars, i)
                    for i in range(threads)
                ]


            patch_slces = [
                target_slice(result_idx, par_vars, i)
                for i in range(threads)
            ]

            with pyrof.timing('  fetch results'):
                patches_fetched = [ray.get(patch) for patch in C_patches]
        with pyrof.timing(' allocate result'):
            C_par = np.empty(target_shape)

        with pyrof.timing(' assignment'):
            for s, patch in zip(patch_slces, patches_fetched):
                C_par[s[0]] = patch

    assert np.array_equal(C, C_par)


# ### Concurrent assignment

def ray_concurr():
    contract_idx = set(x[1]) & set(y[1])
    result_idx = set(x[1] + y[1])


    with pyrof.timing(f'contract simple'):
        C = contract(x,y)
        
    sliced_contract_ray = ray.remote(sliced_contract)
        
    par_vars = [1,3,16, 17]
    threads = 2**len(par_vars)
    target_shape = C.shape

    with pyrof.timing('Ray total compute'):
        with pyrof.timing('Ray compute'):
            with pyrof.timing('  Ray submit'):
                C_patches = [
                    sliced_contract_ray.remote(x, y, par_vars, i)
                    for i in range(threads)
                ]


            patch_slces = [
                target_slice(result_idx, par_vars, i)
                for i in range(threads)
            ]
            
            with pyrof.timing(' allocate result'):
                C_par = np.empty(target_shape)

            idx = 0
            obj_slice_map = {o:s for o, s in zip(C_patches, patch_slces)}
            obj_slice_id = {o:i for o, i in zip(C_patches, range(threads))}
            with pyrof.timing('fetching results'):
                while True:
                    ready_ids, working_ids = ray.wait(C_patches)
                    if len(C_patches)==0:
                        break
                    C_patches = [i for i in C_patches if i not in ready_ids]

                    for patch in ready_ids:
                        sl = obj_slice_map[patch]
                        print(obj_slice_id[patch])
                        C_par[sl[0]] = ray.get(patch)

    assert np.array_equal(C, C_par)


# ## Use unix tools
# ### Threading

# +
from multiprocessing import Pool, Array
from multiprocessing.dummy import Pool as ThreadPool
import os

def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj())


# -

contract_idx = set(x[1]) & set(y[1])
result_idx = set(x[1] + y[1])

with pyrof.timing(f'contract simple'):
    C = contract(x,y)

C_size = sys.getsizeof(C)
print(f'result size: {C_size:e}')
print(f'operands size: {sys.getsizeof(x[0]):e}, {sys.getsizeof(y[0]):e}')
target_shape = C.shape

with pyrof.timing('Total thread contraction time:'):
    par_vars = [1,17, 5]
    threads = 2**len(par_vars)

    os.global_C = np.empty(target_shape)

    def work(i):
        patch = sliced_contract(x, y, par_vars, i)
        sl = target_slice(result_idx, par_vars, i)
        os.global_C[sl[0]] = patch

    pool = ThreadPool(processes=threads)
    print('inited thread pool')
    with pyrof.timing('Thread work'):
        _ = pool.map(work, range(threads))

    C_size = sys.getsizeof(os.global_C)
    print(f'  result size: {C_size:e}')

assert np.array_equal(C, os.global_C)

# ###  Multiprocessing

# +

flat_size = len(C.flatten())
with pyrof.timing('init array'):
    os.global_C = np.empty(target_shape)
    #os.global_C = tonumpyarray(Array('d', flat_size))
#us.global_C = os.global_C.reshape(target_shape)

pool = Pool(processes=threads)
print('inited pool')
with pyrof.timing('parallel work'):
    print('started work')
    _ = pool.map(work, range(threads))

C_size = sys.getsizeof(os.global_C)
print(f'result size: {C_size:e}')
assert np.array_equal(C, os.global_C)

# -


del os.global_C


