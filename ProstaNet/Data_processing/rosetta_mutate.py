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

#Using Rosetta relax to perform single-point mutation
def wild_mutate(start_struct, pdb_chain, variant_resfile, variant):
    rosetta_relax_script_path = os.path.expanduser('./relax.mpi.linuxgccrelease')
    output_directory = os.path.expanduser('./Extra_alpha_mutated')
    input_pdb_path = os.path.expanduser('./Ssym_alpha')
    start_struct_path = os.path.join(input_pdb_path, start_struct)

#The parameters of Rosetta single-point mutation   
    wild_relax_script_arg = [
        os.path.abspath(rosetta_relax_script_path),
        '-in:file:s', os.path.abspath(start_struct_path),
        '-in:file:fullatom',
        '-relax:constrain_relax_to_start_coords',
        '-out:no_nstruct_label', '-relax:ramp_constraints false',
        '-relax:respect_resfile',
        '-packing:resfile', variant_resfile,
        '-default_max_cycles 200',
        '-out:file:scorefile', os.path.join(pdb_chain[0:5] + '_' + variant + '_relaxed.sc'),
        '-out:suffix', '_' + variant + '_relaxed',
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
    base_path = os.path.expanduser('/home/til60/rosetta_src_2021.16.61629_bundle/main/source/bin/TRY/Relax')
    input_pdb_path = os.path.expanduser('%s/Ssym_alpha' %base_path)
    output_directory = os.path.expanduser('%s/Extra_alpha_mutated' %base_path)
    variant_list = os.path.expanduser('%s/Variant_list.txt' %output_directory)

    variants = []
    os.chdir(output_directory)
    with open(variant_list, 'rt') as ipf:
        for l in ipf:
            pdb_chain, pos, w, m = l.strip().split()
            variants.append((pdb_chain, w + pos + m))

    for pdb_chain, variant in variants:
        variant_resfile = pdb_chain + '_' + variant + '.resfile'
        with open(variant_resfile, 'wt') as opf:
            opf.write('NATAA\n')
            opf.write('start\n')
            opf.write(variant[1:-1] + ' ' + pdb_chain[4] + ' PIKAA ' + variant[-1])
        start_struct = os.path.join(input_pdb_path, pdb_chain[0:5] + '.pdb')
        variant_list_path = os.path.join(output_directory, variant_resfile)
        case.append((start_struct, pdb_chain, variant_list_path, variant))
    os.chdir(base_path)

    if use_multiprocessing:
        pool = multiprocessing.Pool( processes = min(max_cpus, multiprocessing.cpu_count()) )

    for args in case:
        if use_multiprocessing:
            pool.apply_async( wild_mutate, args = args )
        else:
            wild_mutate(*args)

    if use_multiprocessing:
        pool.close()
        pool.join()

    d2=datetime.datetime.now()
    print(d2-d1)

