from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import SimpleITK as sitk
import os
import math
import torch
import pytorch_lightning as pl
from torchvision import transforms
import cv2

import monai
from monai.transforms import (
    EnsureChannelFirst,
    ToTensor,
    ScaleIntensityRange,
    RandGaussianNoise,
    AddChanneld,
    Compose,
    RandRotated,
    ScaleIntensityd,    
    ToTensord,
    EnsureType,
    Activations, 
    AsDiscrete, 
    Resized,
    RandZoomd,
    Lambdad,
    CenterSpatialCrop,
    ResizeWithPadOrCrop
)

class HystDataset(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img_path", class_column=None):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column = class_column        

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        vid_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        
        try:            
            vid = self.read_mp4(vid_path)
        except:
            print("Error reading: " + vid_path)
            vid = torch.zeros(10, 256, 256, 3, dtype=torch.float32)

        if(self.transform):
            vid = self.transform(vid)            

        if self.class_column:            
            return vid, torch.tensor(self.df.iloc[idx][self.class_column]).to(torch.long)
        
        return vid

    def read_mp4(self, fname):
        cap = cv2.VideoCapture(fname)
        success = cap.isOpened()
        if not success:
            raise IOError("Error opening the MP4 file")
        frames = []
        while success:
            success,frame = cap.read()
            if frame is not None:
                frames.append(np.expand_dims(frame, axis=0))
        all_frames = np.concatenate(frames, axis=0)        
        return all_frames

class HystDataModule(pl.LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", class_column=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.class_column = class_column        
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = HystDataset(self.df_train, self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.train_transform)
        self.val_ds = HystDataset(self.df_val, self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.valid_transform)
        self.test_ds = HystDataset(self.df_test, self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.valid_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=False)


class TrainTransforms:
    def __init__(self, height: int = 256, num_frames=100):
        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                RandomChoice(num_frames=num_frames),
                EnsureChannelFirst(channel_dim=-1),
                PermuteTimeChannel(),
                transforms.RandomResizedCrop(256, scale=(0.2, 1.0)),                
                ToTensor(dtype=torch.float32),                
                ScaleIntensityRange(0, 255, 0, 1)
            ]
        )
    def __call__(self, inp):
        return self.train_transform(inp)


class EvalTransforms:
    def __init__(self, height: int = 256, num_frames=100):
        self.test_transform = transforms.Compose(
            [
                RandomChoice(num_frames=num_frames),
                EnsureChannelFirst(channel_dim=-1),
                PermuteTimeChannel(),
                ResizeWithPadOrCrop((-1, 256, 256)),
                ToTensor(dtype=torch.float32),                
                ScaleIntensityRange(0, 255, 0, 1)                
            ]
        )
    def __call__(self, inp):
        return self.test_transform(inp)

class TestTransforms:
    def __init__(self, height: int = 256, num_frames=512):
        self.test_transform = transforms.Compose(
            [
                RandomChoice(num_frames=num_frames),
                EnsureChannelFirst(channel_dim=-1),
                PermuteTimeChannel(),
                ResizeWithPadOrCrop((-1, 256, 256)),
                ToTensor(dtype=torch.float32),                
                ScaleIntensityRange(0, 255, 0, 1)                
            ]
        )
    def __call__(self, inp):
        return self.test_transform(inp)


class PermuteTimeChannel:
    def __init__(self, permute=(1,0,2,3)):
        self.permute = permute        
    def __call__(self, x):
        return torch.permute(x, self.permute)

class RandomChoice:
    def __init__(self, num_frames=-1):
        self.num_frames = num_frames
    def __call__(self, x):
        if self.num_frames > 0:
            # idx = torch.randint(0, high=x.shape[0], size=(min(self.num_frames, len(x)),))
            idx = torch.randint(0, high=x.shape[0], size=(self.num_frames,))
            if self.num_frames == 1:
                x = x[idx[0]]
            else:
                idx, _ = torch.sort(idx)
                x = x[idx]
        return x