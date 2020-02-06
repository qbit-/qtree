from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
import sys
sys.path.append('.')
from qtree.simulator import eval_circuit_np

with PyCallGraph(output=GraphvizOutput()):
    eval_circuit_np('inst_2x2_7_0.txt')
