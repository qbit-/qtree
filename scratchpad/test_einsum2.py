import numpy as np
from einsum2 import einsum2
from threading import Thread
import psutil
from time import time, sleep
from contextlib import contextmanager

def threaded(f, daemon=False):
    import queue

    def wrapped_f(q, *args, **kwargs):
        '''this function calls the decorated function and puts the 
        result in a queue'''
        ret = f(*args, **kwargs)
        q.put(ret)

    def wrap(*args, **kwargs):
        '''this is the function returned from the decorator. It fires off
        wrapped_f in a new thread and returns the thread object with
        the result queue attached'''

        q = queue.Queue()

        t = Thread(target=wrapped_f, args=(q,)+args, kwargs=kwargs)
        t.daemon = daemon
        t.start()
        t.result_queue = q
        return t

    return wrap

def threaded_gen(f):
    """ makes the function return a generator, 
    which iterates while thread is running and 
    yields return value of function at last step.
    Usage:
        for res it threaded_gen(f)(*args, **kw):
            #do stuff here
        print('Done threaded exec:',res)
    """

    def wrap(*args, **kwargs):
        thr_f =  threaded(f)
        thread = thr_f(*args, **kwargs)
        while thread.is_alive():
            yield
            sleep(.1)
        thread.join()
        yield thread.result_queue.get()
        return
    return wrap

def proc_count(description='Process count'):
    def decorator(f):
        def f_wrapped(*args, **kwargs):
            thr_gen = threaded_gen(f)
            pnames = set()
            for res in thr_gen(*args, **kwargs):
                names = [proc.name() for proc in psutil.process_iter()]
                names = {name + str(i) for i, name in enumerate(names) if name=='python'}
                pnames = pnames | names
            print(description, len(pnames))
            return res
        return f_wrapped
    return decorator

def cpu_util(description='CPU utilisation:'):
    def decorator(f):
        def f_wrapped(*args, **kwargs):
            thr_gen = threaded_gen(f)
            utils = []
            for res in thr_gen(*args, **kwargs):
                utils.append(psutil.cpu_percent(interval=0))
            print(description, max(utils))
            return res
        return f_wrapped
    return decorator


@contextmanager
def timing(description: str) -> None:
    start = time()
    yield
    ellapsed_time = time() - start
    print(f"{description}: {ellapsed_time}")

@proc_count()
@cpu_util()
def do_einsum(n_dim=4, dim_size=2, einsum=np.einsum):
    shape = [dim_size]*(n_dim + 1)
    idx1 = [0] + list(range(1, n_dim + 1))
    idx2 = [0] + list(range(n_dim+2, 2*(n_dim+1)))
    result_idx = list(set(idx1 + idx2))
    T1 = np.random.randn(*shape)
    T2 = np.random.randn(*shape)

    res = einsum(T1, idx1, T2, idx2, result_idx)
    return res

def profile_einsum(einsum, ndims=range(5, 15), dim_size=2):
    msg_templ = '{ndim} dimensions of size {dim_size} total len: {totlen}'
    for ndim in ndims:
        msg = msg_templ.format(  ndim=ndim
                               , dim_size=dim_size
                               , totlen=dim_size**ndim
                              )
        with timing(msg):
            do_einsum(n_dim=ndim, dim_size=dim_size, einsum=einsum)

def profile_both():
    print('numpy')
    profile_einsum(np.einsum)
    print('einsum2')
    profile_einsum(einsum2)

if __name__=='__main__':
    profile_both()
