from concurrent.futures import ThreadPoolExecutor
import sys
import numpy as np

def main():
    max_workers = 8
    pool = ThreadPoolExecutor(max_workers=max_workers)
    def work(N):
        for i in range(N*N):
            x = np.random.randn(N,N)
            y = np.random.randn(N,N)
            z = np.einsum('ij,jk->ik', x, y)

    K = 1
    N = 1e2
    fs = []
    for i in range(int(K)):
        future = pool.submit(work, int(N))
        fs.append(future)

    print([x.result() for x in fs])

if __name__=="__main__":
    main()

