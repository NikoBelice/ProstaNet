import numpy as np
import os
from tqdm import tqdm
import pathlib
from pathlib import Path
import json
import torch, math
import torch_cluster
import torch.nn.functional as F
import torch_geometric

import biographs as bg
from Bio.PDB.PDBParser import PDBParser

#Convert json files to graphs
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")

ressymbl = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU':'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN':'Q', 'ARG':'R', 'SER': 'S','THR': 'T', 'VAL': 'V', 'TRP':'W', 'TYR': 'Y'}
pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                 'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                'N': 2, 'Y': 18, 'M': 12}

residue_charge = {'C': -0.64, 'H': -0.29, 'N': -1.22, 'Q': -1.22, 'S': -0.80, 'T': -0.80, 'Y': -0.80,
                  'W': -0.79, 'A': -0.37, 'F': -0.37, 'G': -0.37, 'I': -0.37, 'V': -0.37, 'M': -0.37,
                  'P': 0.0, 'L': -0.37, 'E': -1.37, 'D': -1.37, 'K': -0.36, 'R': -1.65}

pcp_dict = {'A':[1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23],
            'C':[1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
            'D':[1.60, 0.11, 2.78, -0.77, 2.95, 0.25, 0.20],
            'E':[1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21],
            'F':[2.94, 0.29, 5.89, 1.79, 5.67, 0.30, 0.38],
            'G':[ 0.00, 0.00, 0.00, 0.00, 6.07, 0.13, 0.15],
            'H':[2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.30],
            'I':[4.19, 0.19, 4.00, 1.80, 6.04, 0.30, 0.45],
            'K':[1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
            'L':[2.59, 0.19, 4.00, 1.70, 6.04, 0.39, 0.31],
            'M':[2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
            'N':[1.60, 0.13, 2.95, -0.60, 6.52, 0.21, 0.22],
            'P':[2.67, 0.00, 2.72, 0.72, 6.80, 0.13, 0.34],
            'Q':[1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25],
            'R':[2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
            'S':[1.31, 0.06, 1.60, -0.04, 5.70, 0.20, 0.28],
            'T':[3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36],
            'V':[3.67, 0.14, 3.00, 1.22, 6.02, 0.27, 0.49],
            'W':[3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42],
            'Y':[2.94, 0.30, 6.47, 0.96, 5.66, 0.25, 0.41]}

Miyazawa = {'A':[-0.12, 0.24, 0.15, 0.27, -0.33, 0.22, 0.38, -0.08, 0.07, -0.37, -0.38, 0.41, -0.27, -0.36, 0.15, 0.1, 0.04, -0.27, -0.2, -0.32],
        'R':[0.24, 0.19, 0.1, -0.24, 0.08, 0.09, -0.22, 0.09, 0.05, 0.0, -0.04, 0.66, 0.03, -0.05, 0.17, 0.16, 0.11, -0.21, -0.25, 0.08],
        'N':[0.15, 0.1, -0.06, 0.02, -0.01, 0.06, 0.12, -0.01, 0.0, 0.14, 0.04, 0.22, 0.04, -0.01, 0.18, 0.09, 0.04, -0.1, -0.11, 0.12],
        'D':[0.27, -0.24, 0.02, 0.29, 0.12, 0.24, 0.44, 0.11, -0.1, 0.22, 0.27, -0.01, 0.3, 0.18, 0.33, 0.1, 0.11, 0.07, -0.07, 0.36],
        'C':[-0.33, 0.08, -0.01, 0.12, -1.19, -0.07, 0.2, -0.31, -0.36, -0.64, -0.65, 0.33, -0.61, -0.67, -0.18, -0.13, -0.15, -0.66, -0.39, -0.59],
        'Q':[0.22, 0.09, 0.06, 0.24, -0.07, 0.2, 0.27, 0.13, 0.15, -0.01, -0.04, 0.28, -0.06, -0.11, 0.17, 0.22, 0.12, -0.02, -0.14, 0.08],
        'E':[0.38, -0.22, 0.12, 0.44, 0.2, 0.27, 0.46, 0.32, 0.0, 0.17, 0.17, -0.06, 0.12, 0.14, 0.37, 0.18, 0.16, 0.0, -0.08, 0.26],
        'G':[-0.08, 0.09, -0.01, 0.11, -0.31, 0.13, 0.32, -0.29, 0.0, -0.13, -0.16, 0.29, -0.17, -0.19, 0.02, -0.01, -0.04, -0.25, -0.22, -0.15],
        'H':[0.07, 0.05, 0.0, -0.1, -0.36, 0.15, 0.0, 0.0, -0.4, -0.13, -0.18, 0.38, -0.29, -0.34, 0.01, 0.04, -0.03, -0.37, -0.3, -0.06],
        'I':[-0.37, 0.0, 0.14, 0.22, -0.64, -0.01, 0.17, -0.13, -0.13, -0.74, -0.81, 0.24, -0.66, -0.73, -0.05, 0.03, -0.15, -0.6, -0.49, -0.67],
        'L':[-0.38, -0.04, 0.04, 0.27, -0.65, -0.04, 0.17, -0.16, -0.18, -0.81, -0.84, 0.22, -0.7, -0.8, -0.12, -0.02, -0.15, -0.62, -0.55, -0.74],
        'K':[0.41, 0.66, 0.22, -0.01, 0.33, 0.28, -0.06, 0.29, 0.38, 0.24, 0.22, 0.76, 0.29, 0.19, 0.47, 0.36, 0.33, 0.09, -0.05, 0.29],
        'M':[-0.27, 0.03, 0.04, 0.3, -0.61, -0.06, 0.12, -0.17, -0.29, -0.66, -0.7, 0.29, -0.7, -0.83, -0.13, 0.05, -0.11, -0.73, -0.56, -0.51],
        'F':[-0.36, -0.05, -0.01, 0.18, -0.67, -0.11, 0.14, -0.19, -0.34, -0.73, -0.8, 0.19, -0.83, -0.88, -0.19, -0.12, -0.15, -0.68, -0.58, -0.67],
        'P':[0.15, 0.17, 0.18, 0.33, -0.18, 0.17, 0.37, 0.02, 0.01, -0.05, -0.12, 0.47, -0.13, -0.19, 0.11, 0.2, 0.13, -0.37, -0.25, -0.05],
        'S':[0.1, 0.16, 0.09, 0.1, -0.13, 0.22, 0.18, -0.01, 0.04, 0.03, -0.02, 0.36, 0.05, -0.12, 0.2, 0.05, 0.04, -0.01, -0.08, 0.04],
        'T':[0.04, 0.11, 0.04, 0.11, -0.15, 0.12, 0.16, -0.04, -0.03, -0.15, -0.15, 0.33, -0.11, -0.15, 0.13, 0.04, 0.03, -0.02, -0.09, -0.07],
        'W':[-0.27, -0.21, -0.1, 0.07, -0.66, -0.02, 0.0, -0.25, -0.37, -0.6, -0.62, 0.09, -0.73, -0.68, -0.37, -0.01, -0.02, -0.64, -0.49, -0.51],
        'Y':[-0.2, -0.25, -0.11, -0.07, -0.39, -0.14, -0.08, -0.22, -0.3, -0.49, -0.55, -0.05, -0.56, -0.58, -0.25, -0.08, -0.09, -0.49, -0.45, -0.38],
        'V':[-0.32, 0.08, 0.12, 0.36, -0.59, 0.08, 0.26, -0.15, -0.06, -0.67, -0.74, 0.29, -0.51, -0.67, -0.05, 0.04, -0.07, -0.51, -0.38, -0.65]}

Micheletti = {'A': [0.1461, -0.2511, 0.3323, -0.1348, -0.0751, 0.5029, -0.2376, -0.3111, -0.2432, -0.2119, 0.0864, 0.1754, -0.1496, 0.5126, -0.5081, -0.1515, -0.0218, -0.9737, -0.0724, 0.3642],
              'R': [-0.2511, 0.9875, -0.6728, 0.1974, -0.6062, -0.121, -0.4586, 0.2466, 0.9985, 0.1034, -0.1302, 0.7273, -0.4676, 0.4855, -0.0067, -0.118, 0.3967, -1.4845, 0.4237, -0.5168],
              'N': [0.3323, -0.6728, -0.1962, 0.7855, 0.6139, 0.4502, -0.3154, -0.1649, 0.8099, 0.2317, -0.0605, 0.6158, 1.8413, 0.3461, 0.3707, 0.6249, -0.5914, -0.3028, -0.6968, -0.104],
              'D': [-0.1348, 0.1974, 0.7855, -0.0531, 0.2278, -0.1466, 0.2194, 0.1528, -0.2501, 0.2659, 0.0585, -0.0642, 0.1491, 0.4899, 0.0755, -0.1609, 0.2193, -0.7832, 0.0182, 0.0092],
              'C': [-0.0751, -0.6062, 0.6139, 0.2278, -0.2544, 0.1387, 0.2791, 0.1847, 5.4553, 0.2965, -0.0196, -0.604, 1.4331, -1.3925, -0.172, 0.1837, 0.262, -3.5239, 0.2585, 0.0296],
              'Q': [0.5029, -0.121, 0.4502, -0.1466, 0.1387, 0.8438, -0.5234, -0.0425, 0.5803, -0.1875, -0.4168, 0.2349, -0.2908, 0.379, 0.0525, -0.9002, 0.1006, 1.2075, -0.5137, 0.0029],
              'E': [-0.2376, -0.4586, -0.3154, 0.2194, 0.2791, -0.5234, 0.6456, -0.0113, -0.7232, 0.7647, -0.0453, -0.9604, 0.3231, -0.1143, 0.5402, 0.2888, 0.0948, -0.9357, 0.3261, 0.1387],
              'G': [-0.3111, 0.2466, -0.1649, 0.1528, 0.1847, -0.0425, -0.0113, 0.099, -0.0951, 0.0446, -0.1538, -0.1308, 0.2339, 0.0189, 0.9071, -0.3528, 0.1084, -1.2366, -0.0737, 0.1995],
              'H': [-0.2432, 0.9985, 0.8099, -0.2501, 5.4553, 0.5803, -0.7232, -0.0951, 0.1314, -0.0476, -0.4529, 0.2934, 3.1785, -0.019, -0.2032, 0.9858, -0.5871, -0.6739, 0.7276, -0.6893],
              'I': [-0.2119, 0.1034, 0.2317, 0.2659, 0.2965, -0.1875, 0.7647, 0.0446, -0.0476, 0.6801, -0.0782, 0.0855, -0.9283, -0.9792, 0.4353, 0.1538, -0.4179, 0.2734, -0.4792, 0.2618],
              'L': [0.0864, -0.1302, -0.0605, 0.0585, -0.0196, -0.4168, -0.0453, -0.1538, -0.4529, -0.0782, -0.0748, 0.2119, -0.2531, -0.2127, -0.5026, 0.1004, 0.377, 1.0659, 0.354, -0.194],
              'K': [0.1754, 0.7273, 0.6158, -0.0642, -0.604, 0.2349, -0.9604, -0.1308, 0.2934, 0.0855, 0.2119, 0.5109, -0.4667, -0.4479, 0.9888, 0.5015, -0.5895, -0.1668, 0.7956, -0.6987],
              'M': [-0.1496, -0.4676, 1.8413, 0.1491, 1.4331, -0.2908, 0.3231, 0.2339, 3.1785, -0.9283, -0.2531, -0.4667, 3.1655, 0.101, -0.8698, 0.2007, -0.219, 98.4886, -0.3258, -0.5331],
              'F': [0.5126, 0.4855, 0.3461, 0.4899, -1.3925, 0.379, -0.1143, 0.0189, -0.019, -0.9792, -0.2127, -0.4479, 0.101, -1.3128, -0.6986, -0.1223, 0.4102, 0.6057, -0.3256, 0.0008],
              'P': [-0.5081, -0.0067, 0.3707, 0.0755, -0.172, 0.0525, 0.5402, 0.9071, -0.2032, 0.4353, -0.5026, 0.9888, -0.8698, -0.6986, -0.3621, -0.3125, 0.5402, 1.3914, 0.0996, 0.1362],
              'S': [-0.1515, -0.118, 0.6249, -0.1609, 0.1837, -0.9002, 0.2888, -0.3528, 0.9858, 0.1538, 0.1004, 0.5015, 0.2007, -0.1223, -0.3125, -0.0802, -0.2393, -0.233, -0.1895, -0.0443],
              'T': [-0.0218, 0.3967, -0.5914, 0.2193, 0.262, 0.1006, 0.0948, 0.1084, -0.5871, -0.4179, 0.377, -0.5895, -0.219, 0.4102, 0.5402, -0.2393, 0.3269, 0.3848, -0.1235, 0.4075],
              'W': [-0.9737, -1.4845, -0.3028, -0.7832, -3.5239, 1.2075, -0.9357, -1.2366, -0.6739, 0.2734, 1.0659, -0.1668, 98.4886, 0.6057, 1.3914, -0.233, 0.3848, 13.1813, 0.3708, -0.1516],
              'Y': [-0.0724, 0.4237, -0.6968, 0.0182, 0.2585, -0.5137, 0.3261, -0.0737, 0.7276, -0.4792, 0.354, 0.7956, -0.3258, -0.3256, 0.0996, -0.1895, -0.1235, 0.3708, -0.7699, -0.2175],
              'V': [0.3642, -0.5168, -0.104, 0.0092, 0.0296, 0.0029, 0.1387, 0.1995, -0.6893, 0.2618, -0.194, -0.6987, -0.5331, 0.0008, 0.1362, -0.0443, 0.4075, -0.1516, -0.2175, 0.1445]}

Acthely = {'A':[-0.591, -1.302, -0.733, 1.57, -0.146],
           'C':[-1.343, 0.465, -0.862, -1.02, -0.255],
           'D':[1.05, 0.302, -3.656, -0.259, -3.242],
           'E':[1.357, -1.453, 1.477, 0.113, -0.837],
           'F':[-1.006, -0.59, 1.891, -0.397, 0.412],
           'G':[-0.384, 1.652, 1.33, 1.045, 2.064],
           'H':[0.336, -0.417, -1.673, -1.474, -0.078],
           'I':[-1.239, -0.547, 2.131, 0.393, 0.816],
           'K':[1.831, -0.561, 0.533, -0.277, 1.648],
           'L':[-1.019, -0.987, -1.505, 1.266, -0.912],
           'M':[-0.663, -1.524, 2.219, -1.005, 1.212],
           'N':[0.945, 0.828, 1.299, -0.169, 0.933],
           'P':[0.189, 2.081, -1.628, 0.421, -1.392],
           'Q':[0.931, -0.179, -3.005, -0.503, -1.853],
           'R':[1.538, -0.055, 1.502, 0.44, 2.897],
           'S':[-0.228, 1.399, -4.76, 0.67, -2.647],
           'T':[-0.032, 0.326, 2.213, 0.908, 1.313],
           'V':[-1.337, -0.279, -0.544, 1.242, -1.262],
           'W':[-0.595, 0.009, 0.672, -2.128, -0.184],
           'Y':[0.26, 0.83, 3.097, -0.838, 1.512]}

num_positional_embeddings = 16

def _rbf(D, D_min=0., D_max=20., D_count=16, device="cpu"):
        D_mu = torch.linspace(D_min, D_max, D_count, device="cpu")
        D_mu = D_mu.view([1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

def _normalize(tensor, dim=-1):
     return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

#Encoding protein features
class json2graph():
    def __init__(self, root):
        self.root = root
        self.raw_path = os.path.join(self.root, 'json_path')
        self.processed_paths = os.path.join(self.root, 'processed_graph')

    def process(self, datalist):
        with open(datalist) as f:
            data = json.load(f)
            for i in data:
                name = i['name']
                print(name)
                pdb = os.path.join('../../Data_example/mutant', f"{name}.pdb")
                pssm_file = os.path.join('../../Data_example/blast_result', f"{name}.pssm")
                with torch.no_grad():
                    coords = torch.as_tensor(i['coord'],device="cpu", dtype=torch.float32)

                    one_hot = self._get_one_hot_symbftrs(i['seq']) #One-hot encoding

                    res_fea = self._get_res_ftrs(i['seq']) #Acthely factor encoding

                    res_energy = self._get_res_energy(i['seq']) #Micheletti potential encoding

                    res_score = self._get_res_score(pdb) #Rosetta scoring function

                    res_pssm = self._get_pssm(pssm_file) #position specific scoring matrix encoding

                    seq = torch.as_tensor([letter_to_num[a] for a in i['seq']], device="cpu", dtype=torch.long)
                    mask = torch.isfinite(coords.sum(dim=(1,2)))
                    coords[~mask] = np.inf
                    X_ca = coords[:, 1]

                    edge_index = torch_cluster.knn_graph(X_ca, k=30)

                    pos_embeddings = self._positional_embeddings(edge_index)
                    
                    E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
                    rbf = _rbf(E_vectors.norm(dim=-1), D_count=16, device="cpu")

                    dihedrals = self._dihedrals(coords)

                    orientations = self._orientations(X_ca)

                    sidechains = self._sidechains(coords)

                    node_s = torch.tensor(np.hstack((one_hot, res_pssm, res_score, res_energy, res_fea, dihedrals)), dtype = torch.float) #Six nodes scalar features/You can manage the features here
                    node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2) #Nodes vector features
                    edge_s = torch.cat([rbf, pos_embeddings], dim=-1) #Two edges scalar features
                    edge_v = _normalize(E_vectors).unsqueeze(-2) #Edge vector features
                    node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))
                data = torch_geometric.data.Data(x=X_ca, seq=seq, name=name,
                                                 node_s=node_s, node_v=node_v,edge_s=edge_s, edge_v=edge_v,
                                                 edge_index=edge_index, mask=mask)
                torch.save(data, self.processed_paths + "/"+ name +'.pt')

    def _positional_embeddings(self, edge_index, num_embeddings=None, period_range=[2, 1000]):
        num_embeddings = num_embeddings or num_positional_embeddings
        d = edge_index[0] - edge_index[1]
        frequency = torch.exp(torch.arange(0, num_embeddings, 2, dtype=torch.float32, device="cpu")
                              * -(np.log(10000.0) / num_embeddings))
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E
    
    def _dihedrals(self, X, eps=1e-7):
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        D = F.pad(D, [1, 2]) 
        D = torch.reshape(D, [-1, 3])

        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features
    
    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)
    
    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec
    
    def _get_one_hot_symbftrs(self, sequence):
        one_hot_symb = np.zeros((len(sequence),len(pro_res_table)))
        row= 0
        for res in sequence:
            col = pro_res_table.index(res)
            one_hot_symb[row][col]=1
            row +=1
        return torch.tensor(one_hot_symb, dtype= torch.float)
    
    def _get_res_ftrs(self, sequence):
        res_ftrs_out = []
        for res in sequence:
            res_ftrs_out.append(Acthely[res])
        res_ftrs_out= np.array(res_ftrs_out)
        return torch.tensor(res_ftrs_out, dtype = torch.float)

    def _get_res_energy(self, sequence):
        res_energy_out = []
        for res in sequence:
            res_energy_out.append(Micheletti[res])
        res_energy_out = np.array(res_energy_out)
        return torch.tensor(res_energy_out, dtype = torch.float)

    def _get_res_score(self, pdb):
        scoring = False
        profile = []
        for line in open(pdb):
            if line.startswith("VRT"):
                scoring = False
            if scoring:
                data = [float(v) for v in line.split()[1:-1]]
                sele = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                sele_data = [data[i] for i in sele]
                profile.append(sele_data)
            if line.startswith("pose"):
               scoring = True
        return torch.tensor(profile, dtype = torch.float)

    def _get_pssm(self, pssm_file):
        with open(pssm_file, 'r') as f:
            pssm = f.readlines()

        data = []
        for line in pssm[3:]:
            if len(line.split()) == 44:
                profile = []
                for v in line.split()[2:22]:
                    f = 1 / (1 + math.exp(-int(v)))
                    f = round(f, 4)
                    profile.append(f)
                data.append(profile)
        return torch.tensor(data, dtype = torch.float)
    
if __name__ == '__main__':

    datalist = '../../Data_example/json_path/structures_list.json'
    json_graphs = json2graph('../../Data_example')
    json_graphs.process(datalist)
