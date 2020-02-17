import kron
import numpy as np

a = np.random.randn(2, 2**4)
b = np.random.randn(2, 2**4)
a = np.asfortranarray(a)
b = np.asfortranarray(b)
print(a,b)
c = kron.kronprod(a,b)

c2 = np.kron(a, b)
print(np.max(c2-c))

print(c)
