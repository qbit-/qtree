import networkx as nx
import subprocess
from src.logger_setup import log

import os


def gen_cnf(filename, g):
    """
    Genarate QuickBB input file for the graph g

    Parameters
    ----------
    filename : str
           Output file name
    g : networkx.Graph
           Undirected graphical model
    """
    v = g.number_of_nodes()
    e = g.number_of_edges()
    log.info(f"generating config file {filename}")
    cnf = "c a configuration of -qtree simulator\n"
    cnf += f"p cnf {v} {e}\n"
    for line in nx.generate_edgelist(g):
        cnf += line.replace("{}", ' 0\n')

    # print("cnf file:",cnf)
    with open(filename, 'w+') as fp:
        fp.write(cnf)


def run_quickbb(cnffile,
                command='./quickbb_64',
                outfile='output/quickbb_out.qbb',
                statfile='output/quickbb_stat.qbb'):
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
    # sh += "--min-fill-ordering " - this option leads to missed nodes!
    sh += "--time 60 "
    # this makes Docker process too slow and sometimes fails
    # sh += f"--outfile {outfile} --statfile {statfile} "
    sh += f"--cnffile {cnffile} "
    log.info("excecuting quickbb: "+sh)
    process = subprocess.Popen(
        sh.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        log.error(error)
    log.info(output)
    # with open(outfile, 'r') as fp:
    #     log.info("OUTPUT:\n"+fp.read())
    # with open(statfile, 'r') as fp:
    #     log.info("STAT:\n"+fp.read())

    return output
