import os 
from glob import glob
import numpy as np
from tqdm import tqdm
import argparse
# Image handling
from torchvision import datasets, transforms

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, RandomSampler, SequentialSampler, ConcatDataset
import wandb 
import pickle
from models import *
from dataset import CustomDataset, get_transforms, CustomDataset_meta
import random
import time
from learning import train, test


def get_configs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--resize', type=int, default=512)
    parser.add_argument('--crop_size', type=int, default=2)
    parser.add_argument("--batch", type = int, default = 24)
    parser.add_argument("--epoch", type = int, default = 100)
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--weight_decay", type = float, default = 5e-4)
    parser.add_argument("--gpu_num", type=str, default = "0" )
    parser.add_argument("--mode", type=str, default = "train")
    parser.add_argument("--model",type=str,default="CNN_CenterCrop_10", choices=['CNN_CenterCrop_10', 'ResNet18_CenterCrop',
                                                                                 'ResNet18',"CNN_Base", "efficient_b0" ,
                                                                                 "mobile_v3","MLP", "CNN_MLP", "Efficient_CenterCrop",
                                                                                 "Mobilenet_CenterCrop"])
    args = parser.parse_args()

    return args

args = get_configs()
# Control Randomness


random_seed = 7
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_num 

# def custom_fn(image, **kwargs):
#     image = cv2.bitwise_not(image)


lr, weight_decay, epochs = args.lr, args.weight_decay, args.epoch
train_batch_size = args.batch
valid_batch_size = 16
test_batch_size = 16


transform_ = get_transforms(args)

train_dataset = CustomDataset("./CGX/CGXREV1MOD1_res400_box2",'./CGX/cgx_train_0.8.csv',  transform_)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size , shuffle=True)


test_dataset = CustomDataset("./CGX/CGXREV1MOD1_res400_box2", './CGX/cgx_test_0.2.csv',  transform_)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size , shuffle=False)

wandb.init(project='samsung')
wandb.run.name = "size:{}/batch:{}".format(args.resize, args.batch)
wandb.config = {'size' : args.resize, 
                'dataset' : args.batch}


#####################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_selection = {
    'CNN_CenterCrop_10': CNN_CenterCrop_10(args),
    'ResNet18_CenterCrop': ResNet18_CenterCrop(args),
    'ResNet18': ResNet18(),
    "CNN_Base" : CNN_Base(args),
    "efficient_b0" : efficient_b0(args),
    "mobile_v3" : mobile_v3(args),
    "MLP" : MLP(args),
    "CNN_MLP" : CNN_MLP(args),
    "Efficient_CenterCrop" : Efficient_CenterCrop(args),
    "Mobilenet_CenterCrop" : Mobilenet_CenterCrop(args)
}

if args.mode == "train":
    
    net = model_selection[args.model]
    net = net.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    cs_scheduler =optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    _, path = train(net, train_dataloader,test_dataloader, criterion, optimizer, cs_scheduler, args, epochs = args.epoch, device=device)
    
    
    net = model_selection[args.model]
    net = net.to(device)
    model_state = torch.load(path)
    net.load_state_dict(model_state)
    criterion = nn.MSELoss().to(device)
    test(net, test_dataloader, criterion,device,args)

# else:
    