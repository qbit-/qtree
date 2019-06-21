"""
Here we put all system-dependent constants
"""
import numpy as np
import tensorflow as tf

QUICKBB_COMMAND = './quickbb/run_quickbb_64.sh'
MAXIMAL_MEMORY = 1e22   # 100000000 64bit complex numbers

NP_ARRAY_TYPE = np.complex64
TF_ARRAY_TYPE = tf.complex64
