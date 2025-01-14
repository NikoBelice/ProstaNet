import torch
import glob
import os
import numpy as np
from Model_GVP_wy import StaGVP
from torch.utils.data import Dataset as Dataset_n
from torch_geometric.data import DataLoader

#The path to structure graphs of data need to be predicted
processed_dir = "../../predict/processed"
#The npy file of unseen mutation data
npy_file = "../../predict/zy_HuJ3v1"

#Create dataset for unseen data
class Predict_Dataset(Dataset_n):
    def __init__(self, npy_file, processed_dir):
      self.npy_ar = np.loadtxt(npy_file, dtype=str)
      self.processed_dir = processed_dir
      self.wild = self.npy_ar[:,0]
      self.mutant = self.npy_ar[:,5]
      self.n_samples = self.npy_ar.shape[0]

    def __len__(self):
       return(self.n_samples)
    
    def __getitem__(self, index):
       wild = os.path.join(self.processed_dir, self.wild[index]+".pt")
       mutant = os.path.join(self.processed_dir, self.mutant[index]+".pt")
       print(mutant)
       wild = torch.load(glob.glob(wild)[0])
       mutant = torch.load(glob.glob(mutant)[0])
       return(wild, mutant)
    
#Prepare the unseen data
dataset = Predict_Dataset(npy_file = npy_file ,processed_dir = processed_dir)
predloader = DataLoader(dataset=dataset, batch_size=1, num_workers=0, shuffle=False)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")
   
model = StaGVP((91, 3), (45, 6), (32, 1), (32, 1), 0.2) #These parameters should be same as the parameters used in Train_GVP.py
   
model.to(device)

def make_prediction(model, predloader):
   model.load_state_dict(torch.load(f"../../Model/GVP_single_best.pth"))
   model.eval()
   predictions = []
   with torch.no_grad():
      for count, (wild, mutant) in enumerate(predloader):
         wild = wild.to(device)
         mutant = mutant.to(device)
         output = model(wild, mutant)
         print(output)
         predictions.extend(output.cpu().numpy())
   return predictions

prediction = make_prediction(model, predloader)
print(prediction)
