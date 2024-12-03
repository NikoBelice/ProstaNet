import os
import torch
import glob
import numpy as np
import math
from torch.utils.data import Dataset as Dataset_n
from torch_geometric.data import DataLoader
from sklearn.model_selection import KFold


#The folder that stores data graph
processed_dir = "../../LTJ_features/processed_GVP4/"

#training data list 
npy_file = "../../Datasets/Multiple_training_cluster.npy"

#Direct mutation data
class dir_Dataset(Dataset_n):
    def __init__(self, npy_file, processed_dir):
      self.npy_ar = np.loadtxt(npy_file, dtype=str)
      self.processed_dir = processed_dir
      self.wild = self.npy_ar[:,0]
      self.mutant = self.npy_ar[:,5]
      self.label = self.npy_ar[:,6].astype(float)
      self.n_samples = self.npy_ar.shape[0]

    def __len__(self):
       return(self.n_samples)
    
    def __getitem__(self, index):
       wild = os.path.join(self.processed_dir, self.wild[index]+".pt")
       mutant = os.path.join(self.processed_dir, self.mutant[index]+".pt")
       wild = torch.load(glob.glob(wild)[0])
       mutant = torch.load(glob.glob(mutant)[0])
       return(wild, mutant, torch.tensor(self.label[index]))
    
class in_Dataset(Dataset_n):
    def __init__(self, npy_file, processed_dir):
      self.npy_ar = np.loadtxt(npy_file, dtype=str)
      self.processed_dir = processed_dir
      self.wild = self.npy_ar[:,5]
      self.mutant = self.npy_ar[:,0]
      self.label = self.npy_ar[:,7].astype(float)
      self.n_samples = self.npy_ar.shape[0]

    def __len__(self):
       return(self.n_samples)
    
    def __getitem__(self, index):
       wild = os.path.join(self.processed_dir, self.wild[index]+".pt")
       mutant = os.path.join(self.processed_dir, self.mutant[index]+".pt")
       wild = torch.load(glob.glob(wild)[0])
       mutant = torch.load(glob.glob(mutant)[0])
       return(wild, mutant, torch.tensor(self.label[index]))

final_pairs = np.loadtxt(npy_file, dtype=str)
size = final_pairs.shape[0]
    
dir_dataset_multiple = dir_Dataset(npy_file = npy_file ,processed_dir= processed_dir)
in_dataset_multiple = in_Dataset(npy_file = npy_file ,processed_dir= processed_dir)
