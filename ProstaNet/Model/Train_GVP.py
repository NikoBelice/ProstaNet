import torch
import math
from Metrics import *
import numpy as np

import torch.nn as nn
from Data_prepare import dir_dataset, in_dataset
from Model_GVP_wy import StaGVP
from torch_geometric.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset

trainsets = []
validatesets = []

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")

print("Datalength")
print(len(dir_dataset))

total_samples = len(dir_dataset)
n_iterations = math.ceil(total_samples/5)

#Training process
def train(model, device, trainloader, optimizer, epoch):
    
    print(f'Training on {len(trainloader)} samples.....')
    model.train()
    loss_func = nn.BCELoss()
    predictions_tr = torch.Tensor()
    labels_tr = torch.Tensor()

    for count,(wild, mutant, label) in enumerate(trainloader):
        wild = wild.to(device)
        mutant = mutant.to(device)
        optimizer.zero_grad()
        output = model(wild, mutant)
        predictions_tr = torch.cat((predictions_tr, output.cpu()), 0)
        labels_tr = torch.cat((labels_tr, label.view(-1,1).cpu()), 0)
        loss = loss_func(output, label.view(-1,1).float().to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

    loss = loss_func(predictions_tr.float(), labels_tr.float())
    loss_r = loss.detach().numpy()
    loss_r = np.round(loss_r, decimals=3)
    loss_r = f'{loss_r:.3f}'
    labels_tr = labels_tr.detach().numpy()
    predictions_tr = predictions_tr.detach().numpy()
    train_A = get_accuracy(labels_tr, predictions_tr, 0.5)
    train_A_r = np.round(train_A, decimals=3)
    print(f'Epoch {epoch-1} / {num_epochs} [==============================] - train_loss : {loss_r} - train_A : {train_A_r}')

#Validation process
def validate(model, device, validateloader):
    model.eval()
    predictions = torch.Tensor()
    labels = torch.Tensor()
    loss_func = nn.BCELoss()
    with torch.no_grad():
      for count, (wild, mutant, label) in enumerate(validateloader):
        wild = wild.to(device)
        mutant = mutant.to(device)
        output = model(wild, mutant)
        predictions = torch.cat((predictions, output.cpu()), 0)
        labels = torch.cat((labels, label.view(-1,1).cpu()), 0)

    loss = loss_func(predictions.float(), labels.float())
    loss_r = loss.detach().numpy()
    loss_r = np.round(loss_r, decimals=3)
    loss_r = f'{loss_r:.3f}'
    labels = labels.detach().numpy()
    predictions = predictions.detach().numpy()
    val_A = get_accuracy(labels, predictions, 0.5)
    val_A_r = np.round(val_A, decimals=3)
    print(f'Epoch {epoch}/ {num_epochs} [==============================] - val_loss : {loss_r} - val_A : {val_A_r}')
    return loss

seed = 42
torch.manual_seed(seed)
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
k=0

n_epochs_stop = 6
epochs_no_improve = 0
early_stop = False

for fold, (train_indices, val_indices) in enumerate(kfold.split(dir_dataset)):
  dir_trainset = torch.utils.data.Subset(dir_dataset, train_indices)
  dir_validateset = torch.utils.data.Subset(dir_dataset, val_indices)

  in_trainset = torch.utils.data.Subset(in_dataset, train_indices)
  in_validateset = torch.utils.data.Subset(in_dataset, val_indices)

  trainset = ConcatDataset([dir_trainset, in_trainset])
  validateset = ConcatDataset([dir_validateset, in_validateset])
   
  trainloader = DataLoader(dataset=trainset, batch_size=2, num_workers=0, shuffle=True, drop_last=True)
  validateloader = DataLoader(dataset=validateset, batch_size=2, num_workers=0, shuffle=True, drop_last=True)


  k+=1
  min_loss = 100
  best_loss = 100
  
  '''
  (91, 3) is input dimension of node, 91 is scalar features dimension, 3 is vector features dimension.
  (45, 6) is hidden dimension. (32, 1) is input dimension of edge, 32 is scaler features and 1 is vector features.
  0.2 is dropout. All these parameters are adjustable
  '''


  model = StaGVP((91, 3), (45, 6), (32, 1), (32, 1), 0.2)
  model.to(device)
  num_epochs = 500
  loss_func = nn.CrossEntropyLoss()
  optimizer =  torch.optim.Adam(model.parameters(), lr= 0.0001, weight_decay= 0.00001)
  
  for epoch in range(num_epochs):
    train(model, device, trainloader, optimizer, epoch+1)
    val_loss = validate(model, device, validateloader)
     
    # Checkpoint
    if(val_loss < best_loss):
      best_loss = val_loss
      best_loss_epoch = epoch
      torch.save(model.state_dict(), f"../../GVPs1{k}.pth") #path to save the model
      print("Model")
      
    if(val_loss < min_loss):
      epochs_no_improve = 0
      min_loss = val_loss
      min_loss_epoch = epoch
    elif val_loss> min_loss :
      epochs_no_improve += 1
    if epoch > 30 and epochs_no_improve >= n_epochs_stop:
      print('Early stopping!' )
      early_stop = True
      break
