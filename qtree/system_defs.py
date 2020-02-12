"""
Here we put all system-dependent constants
"""
import numpy as np
try:
    import tensorflow as tf
    TF_ARRAY_TYPE = tf.complex64
except ImportError as e:
    TF_ARRAY_TYPE = None
    print(repr(e))


#QUICKBB_COMMAND = './quickbb/run_quickbb_64.sh'
QUICKBB_COMMAND = './quickbb/quickbb_64'
MAXIMAL_MEMORY = 1e22   # 100000000 64bit complex numbers

NP_ARRAY_TYPE = np.complex64
