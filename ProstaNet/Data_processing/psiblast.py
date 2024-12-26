import os
import subprocess

fasta_path = '/home/til60/Desktop/Protein_stability/LTJ_features/fasta2'
output_path = '/home/til60/Desktop/Blast/blast_result'
script_path = '/home/til60/Desktop/Blast/bin/psiblast'
db_path = '/home/til60/Desktop/Blast/UniRef/uniref90.fasta'

#Using Psi-blast to generate PSSM features
def run_psiblast(file_path, name):
#The parameters of psi-blast
    psiblast_args = [
        script_path,
        '-query', file_path,
        '-db', db_path,
        '-num_iterations', '3',
        '-evalue', '0.001',
        '-num_threads', '32',
        '-save_pssm_after_last_round',
        '-out_ascii_pssm', os.path.join(output_path, f"{name}.pssm"),
    ]

    #print( ' '.join(psiblast_args) )
    log_path = os.path.join(output_path, 'psi.out')
    outfile = open(log_path, 'w')
    process = subprocess.run(psiblast_args, stdout=outfile, stderr=subprocess.STDOUT, close_fds = True, cwd = output_path)
    #returncode = process.wait()
    outfile.close()

if __name__ == '__main__':
    for file in os.listdir(fasta_path):
        file_path = os.path.join(fasta_path, file)
        name = os.path.splitext(os.path.basename(file))[0]

        run_psiblast(file_path, name)
    print('finish')
