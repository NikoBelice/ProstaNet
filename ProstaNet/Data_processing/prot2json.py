import numpy as np
import os
import json
import biographs as bg
from Bio.PDB.PDBParser import PDBParser

#Convert proteins into json files
ressymbl = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU':'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN':'Q', 'ARG':'R', 'SER': 'S','THR': 'T', 'VAL': 'V', 'TRP':'W', 'TYR': 'Y'}

class prot2json():
    def __init__(self, root):
        self.root = root
        self.raw_paths = os.path.join(self.root, 'mutant')
        self.json_paths = os.path.join(self.root, 'json_path')
        self.fasta_paths = os.path.join(self.root, 'fasta')
    
    def process(self):
        total_coord = []
        for file in os.listdir(self.raw_paths):
            file_path = os.path.join(self.raw_paths, file)

            output = self._get_coord(file_path)
            self._get_fasta(file_path)
            total_coord.append(output)
        outfile = os.path.join(self.json_paths, 'structures_listv4.json')
        json.dump(total_coord, open(outfile, "w"))

#Generate fasta of each protein, which for getting pssm files
    def _get_fasta(self, file):
        sequence =""
        parser = PDBParser()
        structure = parser.get_structure(id, file)
        name = os.path.splitext(os.path.basename(file))[0]
        file_name = f"{name}.fasta"

        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() in ressymbl.keys():
                        sequence = sequence + ressymbl[residue.get_resname()]

        outfile = os.path.join(self.fasta_paths, file_name)
        with open(outfile, 'w') as file:
            file.write(f'>{name}\n{sequence}\n')

#Generate coordinate for each protein 
    def _get_coord(self, file):
        output = {}
        target_atoms = ["N", "CA", "C", "O"]
        output["name"] = os.path.splitext(os.path.basename(file))[0]
        sequence =""
        coord1 = []
        coord2 = []
        coord3 = []

        parser = PDBParser()
        structure = parser.get_structure(id, file)

        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() in ressymbl.keys():
                        sequence = sequence+ ressymbl[residue.get_resname()]
                        output["seq"] = sequence
                    for atom in target_atoms:
                        vector = residue[atom].get_vector()
                        x, y, z = vector
                        coord1 = [round(x, 3), round(y, 3), round(z, 3)]
                        coord2.append(coord1)
                    coord3. append(coord2)
                    coord2 = []
                output["coord"] = coord3

        return output

if __name__ == '__main__':
    prot_json = prot2json('../Data_example')
    prot_json.process()
