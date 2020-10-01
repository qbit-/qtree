"""
Here we put all system-dependent constants
"""
import numpy as np
import os
import shutil
from pathlib import Path
from .logger_setup import log

QTREE_PATH = Path(os.path.abspath(__file__)).parent.parent
THIRDPARTY_PATH = os.path.join(QTREE_PATH, 'thirdparty')

# Check for QuickBB
QUICKBB_COMMAND = shutil.which('run_quickbb_64.sh')
if not QUICKBB_COMMAND:
    QUICKBB_COMMAND = shutil.which('quickbb_64')
if not QUICKBB_COMMAND:
    quickbb_path = os.path.join(
        THIRDPARTY_PATH, 'quickbb', 'run_quickbb_64.sh')
    if os.path.isfile(quickbb_path):
        QUICKBB_COMMAND = quickbb_path
if not QUICKBB_COMMAND:
    log.warn('QuickBB solver is unavailable')
        
# Check for Tamaki solver
try:
    tw_loc = shutil.which('tw-exact')
    if tw_loc:
        TAMAKI_SOLVER_PATH = os.path.dirname(tw_loc)
    else:
        TAMAKI_SOLVER_PATH = None
    if TAMAKI_SOLVER_PATH is None:
        tamaki_solver_path = os.path.join(
            THIRDPARTY_PATH, 'tamaki_treewidth')
        if os.path.isdir(tamaki_solver_path):
            TAMAKI_SOLVER_PATH = tamaki_solver_path
        else:
            raise Exception(f'No path {tamaki_solver_path}')
except Exception as e:
    log.warn(f'Tamaki solver is unavailable: {e}')

MAXIMAL_MEMORY = 1e22   # 100000000 64bit complex numbers
NP_ARRAY_TYPE = np.complex64

try:
    import tensorflow as tf
    try:
        TF_ARRAY_TYPE = tf.complex64
    except AttributeError:
        pass
except ImportError:
    pass

