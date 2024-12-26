import numpy as np
import os
from tqdm import tqdm
import pathlib
from pathlib import Path

import biographs as bg
from Bio.PDB.PDBParser import PDBParser

import torch
import networkx as nx
from transformers import BertModel, BertTokenizer
from torch_geometric.data import Dataset, Data

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")

# Dictionary for getting Residue symbols
ressymbl = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU':'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN':'Q', 'ARG':'R', 'SER': 'S','THR': 'T', 'VAL': 'V', 'TRP':'W', 'TYR': 'Y'}

#Using ProtBert to encode the proteins
class prot2graph():
    def __init__(self, root):
       self.root = root
       self.raw_paths = os.path.join(self.root, 'raw')
       self.processed_paths = os.path.join(self.root, 'processed_GVP1')

    def process(self, tokenizer, model):
        data_list =[]
        count = 0
        for file in os.listdir(self.raw_paths):
           if file.endswith('.pdb'):
            file_path = os.path.join(self.raw_paths, file)
            struct = self._get_structure(file_path)

            seq = self._get_sequence(struct)

            # Node features extraction
            node_feats = self._get_seq_emb(seq, tokenizer, model)

            # Adjacency matrix extraction
            mat = self._get_adjacency(file_path)

            # Edge index extraction
            edge_index = self._get_edgeindex(file_path, mat)
            
            data = Data(x = node_feats, edge_index = edge_index )
            count += 1
            data_list.append(data)
            torch.save(data, self.processed_paths + "/"+ os.path.splitext(os.path.basename(file))[0]+'.pt')

        self.data_prot = data_list
        print(data_list)
        print(count)

    def _get_adjacency(self, file):
        molecule = bg.Pmolecule(file)
        network = molecule.network()
        mat = nx.adjacency_matrix(network)
        m = mat.todense()
        return m
    
    def _get_edgeindex(self, file, adjacency_mat):
        edge_ind = []
        m = self._get_adjacency(file)

        a = np.nonzero(m > 0)[0]
        b = np.nonzero(m > 0)[1]
        edge_ind.append(a)
        edge_ind.append(b)
        return torch.tensor(np.array(edge_ind), dtype= torch.long)
    
    # Use biopython to get structure from a pdb file
    def _get_structure(self, file):
        parser = PDBParser()
        structure = parser.get_structure(id, file)
        return structure
    
    def _get_sequence(self, structure):
        sequence =""
        for model in structure:
          for chain in model:
            for residue in chain:
              if residue.get_resname() in ressymbl.keys():
                  sequence = sequence+ ressymbl[residue.get_resname()]
        return sequence
    
    # Use ProtBERT to extract features from residues
    def _get_seq_emb(self, sequence, tokenizer, model):

       sequence_example = sequence # A protein sequence of length L
       sequence_example = ' '.join(list(sequence_example))

       encoded_input = tokenizer(sequence_example, add_special_tokens=False, return_tensors='pt').to(device)
       output = model(**encoded_input)
       
       return output['last_hidden_state'][:][0].cpu().detach()#.numpy() # Vector with shape [Lx1024]
    

if __name__ == '__main__':
   
   tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False )
   model = BertModel.from_pretrained("Rostlab/prot_bert_bfd").to(device)
   prot_graphs = prot2graph('../LTJ_Test_features')
   prot_graphs.process(tokenizer, model)
