import click
import sys
sys.path.append('.')
from qtree.simulator import eval_circuit_np, eval_circuit_multiamp_np
from qtree.simulator_mproc import eval_circuit_np_parallel_mproc
from qtree.simulator_mpi import eval_circuit_np_parallel_mpi

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
@click.option('-n', '--num-paralllel-vars', 'par_vars', default=1)
def simulate_mpi(filename, par_vars):
    print(filename, par_vars)
    eval_circuit_np_parallel_mpi(filename)


@cli.command()
@click.argument('filename')
@click.option('-n', '--num-paralllel-vars', 'par_vars', default=1)
def simulate_mproc(filename, par_vars):
    print(filename, par_vars)
    eval_circuit_np_parallel_mproc(filename, n_var_parallel=par_vars)

if __name__ == '__main__':
    cli()

