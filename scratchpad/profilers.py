from threading import Thread
from time import time, sleep
from contextlib import contextmanager
import psutil

@contextmanager
def timing(description: str) -> None:
    start = time()
    yield
    ellapsed_time = time() - start
    print(f"{description}: {ellapsed_time}")

def timed(descr):
    def  decor(f):
        def wrapped(*a,**kw):
            with timing(descr):
                return f(*a, **kw)
        return wrapped
    return decor

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

def profile_decorator(profiler):
    """ An abstract decorator factory.
        Takes a function `profiler` and returns a decorator,
        which takes a function to profile, `profilee`,
        starts it in separate thread and
        passes `profiler` a generator that iterates
        while the therad with `profilee` is running.
        Last vaulue of the generator will be a return value of `profilee`

        Usage:
            @profile_decorator
            def cpu_load(gen, output_fmt='cpu_vals='):
                cpus = []
                while ret in gen:
                    cpus.append(get_cpu_util())
                print(output_fmt, cpus)
                return ret

        Returns:
            profiler_kwargs -> callable -> wrapped_callable
    """
    def wrapper(**profiler_kw):
        def decorator(profilee):
            def wrap(*args, **kwargs):
                thr_gen = threaded_gen(profilee)
                gen = thr_gen(*args, **kwargs)
                res = profiler(gen, **profiler_kw)
                return res
            return wrap
        return decorator
    return wrapper


@profile_decorator
def proc_count(gen, description='Process max count:'):
    pnames = set()
    res = None
    for res in gen:
        names = [proc.name() for proc in psutil.process_iter()]
        names = {name + str(i) for i, name in enumerate(names) 
                 if 'python' in name}
        pnames = pnames | names
    print(description, len(pnames))
    return res

def repeat(n):
    def decor(f):
        def wrap(*args, **kwargs):
            for i in range(n):
                x = f(*args, **kwargs)
            return x
        return wrap
    return decor

@profile_decorator
def cpu_util(gen, description='CPU utilisation:'):
    utils = []
    res = None
    for res in gen:
        utils.append(psutil.cpu_percent(interval=0))
    print(description, max(utils))
    return res
