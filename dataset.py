from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, RandomSampler, SequentialSampler, ConcatDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import argparse
# Image handling
from torchvision import datasets, transforms
from PIL import Image
import os
import torchvision.transforms as T
import torch


class CustomDataset(Dataset):
    def __init__(self, data_dir,csv_file, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.image_paths = self.labels_df['image_name'].tolist()
        self.labels = self.labels_df['CD'].tolist()
        self.groups = self.labels_df['group'].tolist()
        self.data_dir = data_dir
        self.transform = transform
        self.image_name = self.labels_df["image_name"].tolist()
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_paths[idx])
        # image = np.array(Image.open(img_path))
        # image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = Image.fromarray(image)
        image_name = self.image_name[idx]
        label = self.labels[idx]
        group = self.groups[idx]

        if self.transform is not None:
            image = self.transform(image)
        

        return image, label, img_path, image_name, group
    
class CustomDataset_meta(Dataset):
    def __init__(self, data_dir,csv_file, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.image_paths = self.labels_df['image_name'].tolist()
        self.labels = self.labels_df['CD'].tolist()
        self.groups = self.labels_df['group'].tolist()
        self.meta = self.labels_df.iloc[:,12:]
        self.data_dir = data_dir
        self.transform = transform
        self.image_name = self.labels_df["image_name"].tolist()
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_paths[idx])
        # image = np.array(Image.open(img_path))
        # image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = Image.fromarray(image)
        image_name = self.image_name[idx]
        meta = self.meta.iloc[idx,:]
        meta = np.array(meta,dtype=np.float32)
        meta = torch.from_numpy(meta)
        label = self.labels[idx]
        group = self.groups[idx]

        if self.transform is not None:
            image = self.transform(image)
        

        return image, label, img_path, group, meta
    
      
    
#     return image

def get_transforms(args):
    transform_ = transforms.Compose([
        # T.Resize((args.resize,args.resize)),
        # A.Lambda(name='Lambda', image = custom_fn),
        T.ToTensor()
    ])
    
    return transform_

