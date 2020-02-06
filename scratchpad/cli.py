import click
import sys
sys.path.append('.')
from qtree.simulator import eval_circuit_np

@click.command()
@click.argument('filename')
def simulate(filename):
    print(filename)
    eval_circuit_np(filename)

if __name__ == '__main__':
    simulate()



