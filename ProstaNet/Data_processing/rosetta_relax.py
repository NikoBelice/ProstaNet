#!/usr/bin/env python3

from __future__ import print_function

import socket
import sys
import os
import subprocess
import datetime
import pandas as pd
from argparse import ArgumentParser

d1 = datetime.datetime.now()

use_multiprocessing = True
if use_multiprocessing:
    import multiprocessing
    max_cpus = 40

#Using Rosetta relax to relax the wild-type proteins
def wild_relax(start_struct, pdb_chain):
    rosetta_relax_script_path = os.path.expanduser('./relax.mpi.linuxgccrelease')
    output_directory = os.path.expanduser('./Multiple_alpha/multiple')
    input_pdb_path = os.path.expanduser('./Multiple_alpha')
    start_struct_path = os.path.join(input_pdb_path, start_struct)

#The parameters of Rosetta relax   
    wild_relax_script_arg = [
        os.path.abspath(rosetta_relax_script_path),
        '-in:file:s', os.path.abspath(start_struct_path),
        '-in:file:fullatom',
        '-relax:constrain_relax_to_start_coords',
        '-out:no_nstruct_label', '-relax:ramp_constraints false',
        '-default_max_cycles 200',
        '-out:file:scorefile', os.path.join(pdb_chain + '_relaxed.sc'),
        '-out:suffix', '_relaxed',
    ]

    log_path = os.path.join(output_directory, 'rosetta.out')

    print( 'Running Rosetta with args:' )
    print( ' '.join(wild_relax_script_arg) )
    print( 'Output logged to:', os.path.abspath(log_path) )
    print()

    outfile = open(log_path, 'w')
    process = subprocess.Popen(wild_relax_script_arg , stdout=outfile, stderr=subprocess.STDOUT, close_fds = True, cwd = output_directory)
    returncode = process.wait()
    outfile.close()

if __name__=='__main__':

    case = []
    input_pdb_path = os.path.expanduser('./Multiple_alpha')

    for start_struct in os.listdir(input_pdb_path):
        pdb_chain = os.path.splitext(start_struct)[0]
        case.append((start_struct, pdb_chain))

    if use_multiprocessing:
        pool = multiprocessing.Pool( processes = min(max_cpus, multiprocessing.cpu_count()) )

    for args in case:
        if use_multiprocessing:
            pool.apply_async( wild_relax, args = args )
        else:
            wild_relax(*args)

    if use_multiprocessing:
        pool.close()
        pool.join()

    d2=datetime.datetime.now()
    print(d2-d1)

