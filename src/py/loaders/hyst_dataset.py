from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import torch
import pytorch_lightning as pl
from torchvision import transforms
import cv2
import imageio
import random

from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler

from monai.transforms import (
    EnsureChannelFirst,
    ToTensor,
    ScaleIntensityRange,
    RandGaussianNoise,
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
import pdb

class HystDataset(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img_path", len_column = 'len_video', class_column=None, num_frames=30):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column = class_column
        self.len_column = len_column
        self.num_frames = num_frames

        # self.df = self.df.loc[self.df['length_video'] > 30]

        self.img_size = 256

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        # print("Worker:", torch.utils.data.get_worker_info(),"Grabbing: ", idx )
        vid_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        try:
            vid = self.read_mp4_cv2(vid_path)
            # vid = self.read_and_select_mp4_frames_cv2(vid_path, self.num_frames)
        except:
            print("Error reading: " + vid_path)
            vid = torch.zeros(self.num_frames, self.img_size, self.img_size, 3, dtype=torch.float32)

        if(self.transform):
            vid = self.transform(vid)

        if self.class_column:
            return vid, torch.tensor(self.df.iloc[idx][self.class_column]).to(torch.long)
        
        return  vid
    
    def read_mp4_imageio(self, fname):
        
        video_reader = imageio.get_reader(fname)
        frames = []
        for frame in video_reader:
            if frame is not None:
                frame = cv2.resize(frame, (self.img_size, self.img_size)) 
                frames.append(np.expand_dims(frame, axis=0))

        all_frames = np.concatenate(frames, axis=0)
        video_reader.close()
        
        return all_frames

    def read_mp4_cv2(self, fname):
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

        cap.release()
        cv2.destroyAllWindows()

        return all_frames

    def read_and_select_mp4_frames_cv2(self, vid_path, num_frames):
        cap = cv2.VideoCapture(vid_path)
        step = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < num_frames:
            return torch.zeros(self.num_frames, self.img_size, self.img_size, 3, dtype=torch.float32)

        start_idx = max(0,total_frames - num_frames*step)
        if start_idx !=0:
            start_idx = np.random.randint(0,start_idx, (1,))


        end_idx = start_idx + num_frames*step
        idx = torch.arange(int(start_idx), int(end_idx), step)

        if end_idx > total_frames:
            end_idx = min(start_idx + num_frames, total_frames)
            idx = torch.arange(int(start_idx), int(end_idx))

        frames = []
        for frame_number in idx:

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            success, frame = cap.read()
            if success:
                frames.append(frame)
            else:
                frames.append(np.zeros_like(frames[-1]))
            
        video = np.stack(frames, axis=0)
        
        cap.release()
        cv2.destroyAllWindows()

        return video


class HystDataModule(pl.LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path",
                 num_frames = None, class_column=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
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
        self.num_frames = num_frames

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = HystDataset(self.df_train, self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.train_transform, num_frames=self.num_frames)
        self.val_ds = HystDataset(self.df_val, self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.valid_transform, num_frames=self.num_frames)
        self.test_ds = HystDataset(self.df_test, self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.valid_transform, num_frames=self.num_frames)

    def train_dataloader(self):

        # distributed_sampler = DistributedSampler(self.df_train, shuffle=True)
        # balanced_batch_sampler = BalancedBatchSampler(self.df_train, indices=distributed_sampler, batch_size=self.batch_size,  class_column=self.class_column, drop_last=False)
        # train_dataloader = DataLoader(self.train_ds, num_workers=self.num_workers, batch_sampler=balanced_batch_sampler, pin_memory=False)
        
        train_dataloader = DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False, drop_last=self.drop_last, shuffle=True)

        return train_dataloader

    def val_dataloader(self):
        # distributed_sampler = DistributedSampler(self.df_val, shuffle=True)
        # balanced_batch_sampler = BalancedBatchSampler(self.df_val, indices=distributed_sampler, batch_size=self.batch_size,  class_column=self.class_column)
        # val_dataloader = DataLoader(self.val_ds, num_workers=self.num_workers, batch_sampler=balanced_batch_sampler, pin_memory=False)

        val_dataloader = DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False, drop_last=self.drop_last, shuffle=False)

        return val_dataloader

    def test_dataloader(self):

        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False, drop_last=self.drop_last, shuffle=False)


class TrainTransforms:
    def __init__(self, height: int = 256, num_frames=30):
        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                RandomChoice(num_frames=num_frames),
                EnsureChannelFirst(channel_dim=-1),
                PermuteTimeChannel(),
                transforms.RandomResizedCrop(height, scale=(0.2, 1.0),antialias=True),                
                ToTensor(dtype=torch.float32),                
                ScaleIntensityRange(0, 255, 0, 1)
            ]
        )
    def __call__(self, inp):
        return self.train_transform(inp)

class EvalTransforms:
    def __init__(self, height: int = 256, num_frames=30):
        self.test_transform = transforms.Compose(
            [
                RandomChoice(num_frames=num_frames),
                EnsureChannelFirst(channel_dim=-1),
                PermuteTimeChannel(),
                ResizeWithPadOrCrop((-1, height, height)),
                ToTensor(dtype=torch.float32),                
                ScaleIntensityRange(0, 255, 0, 1)                
            ]
        )
    def __call__(self, inp):
        return self.test_transform(inp)

class TestTransforms:
    def __init__(self, height: int = 256, num_frames=30):
        self.test_transform = transforms.Compose(
            [
                RandomChoice(num_frames=num_frames),
                EnsureChannelFirst(channel_dim=-1),
                PermuteTimeChannel(),
                ResizeWithPadOrCrop((-1, height, height)),
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
            idx = torch.randint(0, high=x.shape[0], size=(self.num_frames,))
            if self.num_frames == 1:
                x = x[idx[0]]
            else:
                idx, _ = torch.sort(idx)
                x = x[idx]
        return x 



class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, indices:DistributedSampler = None, batch_size=4, class_column='class',  drop_last=False):

        self.dataset = dataset
        self.class_column = class_column
        self.n_classes = 2 #len(np.unique(self.dataset[self.class_column]))
        self.n_samples = max(1, (batch_size // self.n_classes))
        self.batch_size = batch_size
        self.drop_last = drop_last

        # self.num_iterations = (len(self.dataset)) // self.batch_size
        self.num_iterations = 16

        if indices is not None:
            self.indices = indices
        else:
            self.indices= list(range(len(dataset)))

        self.class_indices = [[] for _ in range(self.n_classes)]

        # list of indices per class
        for idx in range(len(dataset)):
            label = dataset.iloc[idx][self.class_column]
            self.class_indices[label].append(idx)


        # repeating indices if number of samples per class is < n_samples
        for i in range(self.n_classes):
            if len(self.class_indices[i]) <= self.n_samples:
                repeats = self.n_samples // len(self.class_indices[i]) + 1
                self.class_indices[i] = (self.class_indices[i] * repeats)

        
        for i in range(self.n_classes):
            random.shuffle(self.class_indices[i])

    def __iter__(self):
        it = 0
        import copy

        copy_indexes = copy.deepcopy(self.class_indices)

        while it < self.num_iterations:   
            batch_indices = []
        
            for i in range(self.n_classes):
                j =0
                class_indexes = copy_indexes[i]
                while j < self.n_samples:
                    idx = np.random.randint(0, len(class_indexes))

                    batch_indices.append(class_indexes[idx])
                    # class_indexes.pop(idx)
                    j+=1
            it +=1
            yield batch_indices
            batch_indices = []

    def __len__(self):
        return self.num_iterations 
