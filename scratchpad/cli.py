import click
import sys
sys.path.append('.')
from qtree.simulator import eval_circuit_np, eval_circuit_multiamp_np
from qtree.simulator_mproc import eval_circuit_np_parallel_mproc

@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    click.echo('Debug mode is %s' % ('on' if debug else 'off'))

@cli.command()
@click.argument('filename')
def simulate_per_amp(filename):
    print(filename)
    eval_circuit_np(filename)

@cli.command()
@click.argument('filename')
def simulate(filename):
    print(filename)
    eval_circuit_multiamp_np(filename)

@cli.command()
@click.argument('filename')
def simulate_mproc(filename):
    print(filename)
    eval_circuit_np_parallel_mproc(filename)

if __name__ == '__main__':
    cli()



