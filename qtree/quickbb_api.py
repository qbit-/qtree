"""
This module implements interface to QuickBB program.
QuickBB is quite cranky to its input
"""
import networkx as nx
import subprocess
import os

import qtree.system_defs as defs
from qtree.logger_setup import log


def gen_cnf(filename, graph):
    """
    Generate QuickBB input file for the graph.
    This function ALWAYS expects a simple Graph (not MultiGraph)
    without self loops,
    because QuickBB does not understand these situations.

    Parameters
    ----------
    filename : str
           Output file name
    graph : networkx.Graph
           Undirected graphical model
    """
    v = graph.number_of_nodes()
    e = graph.number_of_edges()
    log.info(f"generating config file {filename}")
    cnf = "c a configuration of -qtree simulator\n"
    cnf += f"p cnf {v} {e}\n"

    # Convert possible MultiGraph to Graph (avoid repeated edges)
    for edge in graph.edges():
        u, v = edge
        # print only if this is not a self-loop
        # Node numbering in QuickBB is 1-based
        if u != v:
            cnf += '{} {} 0\n'.format(int(u)+1, int(v)+1)

    # print("cnf file:",cnf)
    with open(filename, 'w+') as fp:
        fp.write(cnf)


def run_quickbb(cnffile,
                command=defs.QUICKBB_COMMAND,
                outfile='output/quickbb_out.qbb',
                statfile='output/quickbb_stat.qbb',
                extra_args=" --min-fill-ordering --time 60 "):
    """
    Run QuickBB program and collect its output

    Parameters
    ----------
    cnffile : str
         Path to the QuickBB input file
    command : str, optional
         QuickBB command name
    outfile : str, optional
         QuickBB output file
    statfile : str, optional
         QuickBB stat file
    extra_args : str, optional
         Optional commands to QuickBB. Default: --min-fill-ordering --time 60
    Returns
    -------
    output : str
         Process output
    """
    # try:
    #     os.remove(outfile)
    #     os.remove(statfile)
    # except FileNotFoundError as e:
    #     log.warn(e)
    #     pass
    sh = command + " "
    # this makes Docker process too slow and sometimes fails
    # sh += f"--outfile {outfile} --statfile {statfile} "
    if extra_args is not None:
        sh += extra_args
    sh += f"--cnffile {cnffile} "

    log.info("excecuting quickbb: "+sh)
    process = subprocess.Popen(
        sh.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        log.error(error)
        print(error)
    # log.info(output)
    # with open(outfile, 'r') as fp:
    #     log.info("OUTPUT:\n"+fp.read())
    # with open(statfile, 'r') as fp:
    #     log.info("STAT:\n"+fp.read())

    return output
