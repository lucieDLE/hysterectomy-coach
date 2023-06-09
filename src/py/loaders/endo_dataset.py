from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import SimpleITK as sitk
import nrrd
import os
import math
import torch
import pytorch_lightning as pl
from torchvision import transforms

from pl_bolts.transforms.dataset_normalizations import (
    imagenet_normalization
)

import monai
from monai.transforms import (
    AsChannelFirst,
    ToTensor,
    ScaleIntensityRange,
    RandGaussianNoise,
    AddChanneld,
    AsChannelFirstd,
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
)

class EndoDataset(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img_path", class_column=None):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column = class_column        

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        
        try:
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        except:
            print("Error reading: " + img_path)
            img = torch.zeros(3, 256, 256, dtype=torch.float32)

        if(self.transform):
            img = self.transform(img)

        if self.class_column:            
            return img, torch.tensor(self.df.iloc[idx][self.class_column]).to(torch.long)
        
        return img

# class TTDatasetStacks(Dataset):
#     def __init__(self, df, mount_point = "./", img_column='img_path', class_column=None, transform=None):
#         self.df = df
#         self.mount_point = mount_point        
#         self.transform = transform
#         self.img_column = img_column
#         self.class_column = class_column        

#     def __len__(self):
#         return len(self.df.index)

#     def __getitem__(self, idx):
        
#         img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

#         try:
#             # img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
#             img, head = nrrd.read(img_path, index_order="C")                        
#             img = torch.tensor(img, dtype=torch.float32)
#             img = img.permute((0, 3, 1, 2))
#         except:
#             print("Error reading stacks: " + img_path)            
#             img = torch.zeros(16, 3, 448, 448, dtype=torch.float32)

#         img = (img/255.0)

#         if self.transform:
#             img = self.transform(img)

#         if self.class_column:
#             return img, torch.tensor(self.df.iloc[idx][self.class_column]).to(torch.long)

#         return img

# class TTDataModuleSeg(pl.LightningDataModule):
#     def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", seg_column="seg_path", train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
#         super().__init__()

#         self.df_train = df_train
#         self.df_val = df_val
#         self.df_test = df_test
#         self.mount_point = mount_point
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.img_column = img_column
#         self.seg_column = seg_column        
#         self.train_transform = train_transform
#         self.valid_transform = valid_transform
#         self.test_transform = test_transform
#         self.drop_last=drop_last

#     def setup(self, stage=None):

#         # Assign train/val datasets for use in dataloaders

#         self.train_ds = monai.data.Dataset(data=TTDatasetSeg(self.df_train, mount_point=self.mount_point, img_column=self.img_column, seg_column=self.seg_column), transform=self.train_transform)

#         self.val_ds = monai.data.Dataset(TTDatasetSeg(self.df_val, mount_point=self.mount_point, img_column=self.img_column, seg_column=self.seg_column), transform=self.valid_transform)
#         self.test_ds = monai.data.Dataset(TTDatasetSeg(self.df_test, mount_point=self.mount_point, img_column=self.img_column, seg_column=self.seg_column), transform=self.test_transform)

#     def train_dataloader(self):
#         return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

#     def val_dataloader(self):
#         return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

#     def test_dataloader(self):
#         return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)


class EndoDataModule(pl.LightningDataModule):
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
        self.train_ds = EndoDataset(self.df_train, self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.train_transform)
        self.val_ds = EndoDataset(self.df_val, self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.valid_transform)
        self.test_ds = EndoDataset(self.df_test, self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.valid_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)


class TrainTransforms:

    def __init__(self, height: int = 256):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                AsChannelFirst(),
                ToTensor(dtype=torch.float32),
                ScaleIntensityRange(0, 255, 0, 1),
                RandGaussianNoise(prob=0.25),
                # transforms.RandomResizedCrop(height, scale=(0.2, 1.0)),
                # transforms.RandomApply([transforms.ColorJitter(brightness=[.5, 1.8], contrast=[0.5, 1.8], saturation=[.5, 1.8], hue=[-.2, .2])], p=0.5),
                # transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(degrees=90),
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)


class EvalTransforms:

    def __init__(self, height: int = 256):

        self.test_transform = transforms.Compose(
            [
                AsChannelFirst(),
                ToTensor(dtype=torch.float32),
                ScaleIntensityRange(0, 255, 0, 1),
                transforms.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        return self.test_transform(inp)



class EvalTransformsRGB:

    def __init__(self, height: int = 256):

        self.test_transform = transforms.Compose(
            [
                AsChannelFirst(),
                ToTensor(dtype=torch.float32),                
                transforms.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        return self.test_transform(inp)

# class TrainTransformsSeg:
#     def __init__(self):
#         # image augmentation functions
#         color_jitter = transforms.ColorJitter(brightness=[.5, 1.8], contrast=[0.5, 1.8], saturation=[.5, 1.8], hue=[-.2, .2])
#         self.train_transform = Compose(
#             [
#                 AsChannelFirstd(keys=["img"]),
#                 AddChanneld(keys=["seg"]),
#                 Resized(keys=["img", "seg"], spatial_size=[512, 512], mode=['area', 'nearest']),
#                 # RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=2),
#                 RandRotated(keys=["img", "seg"], prob=0.5, range_x=math.pi/2.0, range_y=math.pi/2.0, mode=["bilinear", "nearest"], padding_mode='reflection'),
#                 RandZoomd(keys=["img", "seg"], prob=0.5, min_zoom=0.8, max_zoom=1.5, mode=["area", "nearest"], padding_mode='reflect'),
#                 ScaleIntensityd(keys=["img"]),                
#                 Lambdad(keys=['img'], func=lambda x: color_jitter(x))
#             ]
#         )
#     def __call__(self, inp):
#         return self.train_transform(inp)

# class EvalTransformsSeg:
#     def __init__(self):
#         self.eval_transform = Compose(
#             [
#                 AsChannelFirstd(keys=["img"]),
#                 AddChanneld(keys=["seg"]),                
#                 Resized(keys=["img", "seg"], spatial_size=[512, 512], mode=['area', 'nearest']),
#                 ScaleIntensityd(keys=["img"])                
#             ]
#         )

#     def __call__(self, inp):
#         return self.eval_transform(inp)