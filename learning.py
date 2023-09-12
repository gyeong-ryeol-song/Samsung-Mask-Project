import os 
from glob import glob
from telnetlib import X3PAD
from tkinter import image_names
import numpy as np
import pandas as pd
from tqdm import tqdm

# Image handling
from torchvision import datasets, transforms
from PIL import Image

# Pytorch
import torch

from torch.nn import functional as F

import wandb 
import torchvision.transforms as T

import time


def train(net, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, args, epochs=10, device='cpu'):
    start = time.time()
    print('Training for {} epochs on {}'.format(epochs, device))
    
    train_loss_list = []
    val_loss_list = []
    best_loss = 10000000000.
    best_epoch = 0
    path = None
    for epoch in range(1,epochs+1):
        print("Epoch {}/{}".format(epoch, epochs))
        
        net.train()  # put network in train mode for Dropout and Batch Normalization
        train_loss = 0.0#torch.tensor(0., device=device)  # loss and accuracy tensors are on the GPU to avoid data transfers

        for X, y, image_path, image_name, group in tqdm(train_dataloader):
            
            # X = X[:, 0, :, :].unsqueeze(dim=1)
            
            X = X.to(device)
    
            y = y.to(device)
            # import pdb;pdb.set_trace()
            optimizer.zero_grad()

            preds = net(X)
            # import pdb;pdb.set_trace()
            # import pdb;pdb.set_trace()
            preds = preds.squeeze()
        
            #preds = preds.to(torch.float64)
            # import pdb; pdb.set_trace()
            loss = criterion(preds.to(torch.float32), y.to(torch.float32))
            
            #import pdb
            #pdb.set_trace() 

            # loss = F.mse_loss(preds, y)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                tmp =  loss.detach().cpu().numpy() 
                train_loss += tmp #loss.detach().cpu().numpy() 
                #print( 'train ' , type(train_loss) , train_loss)
            #break 


        
        # validation phase
        if valid_dataloader is not None:
            net.eval()  # put network in train mode for Dropout and Batch Normalization
            valid_loss = 0.0  #torch.tensor(0., device=device)

            with torch.no_grad():
                for X, y, image_path, image_name,group  in tqdm(valid_dataloader):
 
                    X = X[:, 0, :, :].unsqueeze(dim=1)
                    X = X.to(device)
                
                    y = y.to(device)
                    preds = net(X).float()
                    preds = preds.squeeze()
                    loss = criterion(preds, y)
                    tmp =  loss.detach().cpu().numpy() 
                    valid_loss += tmp
        if scheduler is not None: 
            scheduler.step()

        
        current_train_loss = train_loss/len(train_dataloader.dataset)
        current_val_loss = valid_loss/len(valid_dataloader.dataset)
        train_loss_list.append(current_train_loss)
        print('Training loss: {}\n'.format(current_train_loss))
        
        if valid_dataloader is not None:
            val_loss_list.append(current_val_loss)
            print('Valid loss: {}\n'.format(current_val_loss))
            print("best_epoch: {}".format(best_epoch))
            
        wandb.log({"train_loss " : current_train_loss,
                    "valid_loss : ": current_val_loss,})
        
        if best_loss > current_val_loss :
            best_loss = current_val_loss
            best_epoch = epoch
            print("best_epoch: {}".format(best_epoch))
            print("model saved\n")
            path = "./save_model/{}_size_{}_batch_{}_{}_epoch_{}_.pt".format(args.model, args.resize,args.batch, args.lr, args.epoch)
            torch.save(net.state_dict(), path)


    end = time.time()
    print('Total training time: {} seconds\n'.format(end-start))
    return net, path

def test(net, test_loader, criterion,device,args):
    net.eval()  # put network in train mode for Dropout and Batch Normalization
    test_loss = 0.0    #torch.tensor(0., device=device)
    test_loss_list= []

    # valid_accuracy = torch.tensor(0., device=device)
    with torch.no_grad():
        for X, y, image_path, image_names, group  in tqdm(test_loader):
            X = X[:, 0, :, :].unsqueeze(dim=1)
            X = X.to(device)
        
            y = y.to(device)
            preds = net(X)
            loss = criterion(preds, y)
            # loss = F.mse_loss(preds, y)

            tmp = loss.detach().cpu().numpy()
            test_loss += tmp    # loss # * test_dataloader.batch_size

            #import pdb
            #pdb.set_trace() 

            path = np.array(image_path)
            group = np.array(group)
            preds = preds.detach().cpu().numpy().squeeze()
            y = y.detach().cpu().numpy() 

            tmp = np.stack([path, group , preds, y], 1)
            
            try:
                test_loss_list =  np.concatenate((test_loss_list, tmp), axis=0 ) 
            except:
                test_loss_list = tmp 
            print(preds, y)
            #break

    np.save('./save_model/{}_size_{}_batch_{}_{}_epoch_{}.npy'.format(args.model, args.resize, args.batch, args.lr, args.epoch), np.array(test_loss_list))
    print('test  loss: {}\n'.format(test_loss/len(test_loader.dataset)))

