from multiprocessing import Pool
import sys
import numpy as np

def work(N):
    for i in range(N*N):
        x = np.random.randn(N,N)
        y = np.random.randn(N,N)
        z = np.matmul(x,y)
        #z = np.einsum('ij,jk->ik', x, y)
    print ('done')

def main():
    max_workers = 32
    pool = Pool(max_workers)

    K = 63
    N = 5*1e2
    fs = []
    for i in range(int(K)):
        future = pool.apply_async(work, (int(N),) )
        fs.append(future)

    print([x.get() for x in fs])

if __name__=="__main__":
    main()

