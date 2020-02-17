import time
import numpy
from wrapper import w_fortranmodule

N = 10**7
def main():
    xs = numpy.linspace(1, 10, N)
    tstart = time.time()
    ys = w_fortranmodule.w_test(xs)
    dt = time.time() - tstart
    print('Time elapsed: %.5fs' % dt)

if __name__ == '__main__':
    main()
