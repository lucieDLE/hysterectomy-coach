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
import SimpleITK as sitk
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
import monai
import math
from monai.transforms import (
    EnsureChannelFirst,
    EnsureChannelFirstd,
    ToTensor,
    SpatialPad,
    ScaleIntensityRange,
    RandGaussianNoise,
    Compose,
    RandRotated,
    ScaleIntensityd,    
    ToTensord,
    EnsureType,
    Activations, 
    AsDiscrete, 
    Padd,
    Resized,
    RandZoomd,
    Lambdad,
    RandFlipd,
    CenterSpatialCrop,
    ResizeWithPadOrCrop
)
import albumentations as A
from transformers import Mask2FormerImageProcessor

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

class HystDatasetSeg(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img_path", seg_column = 'seg_path', class_column='class_column'):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.seg_column = seg_column
        self.class_column = class_column

        self.df_subject = self.df[self.img_column].drop_duplicates().reset_index()

    def __len__(self):
        return len(self.df_subject)

    def __getitem__(self, idx):

        subject = self.df_subject.iloc[idx][self.img_column] # frame path
        img_path = os.path.join(self.mount_point, subject)

        df_patches = self.df.loc[ self.df[self.img_column] == subject]

        img_t = torch.tensor(np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(img_path)).copy())).to(torch.float32)
        img_t /= 255
        shape = img_t.shape[:2]
        self.current_img = img_path

        bbx = []
        masks = []
        labels = []

        ## create targets for mask rcnn model
        for j, row in df_patches.iterrows():

            label = row[self.class_column]
            labels.append(torch.tensor(label).unsqueeze(0))

            seg_path = os.path.join(self.mount_point, row[self.seg_column])
            seg_t = torch.tensor(np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).copy())).to(torch.float32)

            if row['w']*row['h'] < 0.05:
                # print('bbx too small; removing')
                pass

            x,y,w,h = row['x']*shape[1], row['y']*shape[0], row['w']*shape[1], row['h']*shape[0]
            bb = torch.tensor([np.clip(x,0,shape[1]-5), 
                               np.clip(y,0,shape[0]-5), 
                               np.clip((x+w+1),5,shape[1]), 
                               np.clip((y+h+1),5,shape[0])])
            
            bbx.append(bb.unsqueeze(0))
            masks.append(seg_t.unsqueeze(0))
        
        masks = torch.cat(masks)
        bbx = torch.cat(bbx, axis=0)
        labels = torch.cat(labels)
        
        d_augmented = self.transform(image=img_t.numpy(), bboxes=bbx.numpy(), category_ids=labels.numpy(), mask=masks.numpy())
        img_t = d_augmented['image']
        masks = d_augmented['mask']
        boxes = d_augmented['bboxes']
        d = {"img": torch.tensor(img_t).permute(2,0,1), "labels": labels, "masks":  torch.tensor(masks), "boxes":torch.tensor(boxes), 'name':self.current_img}
    
        return  d
  
class HystDatasetFormer(Dataset):
    def __init__(self, df, mount_point="./", transform=None, img_column="img_path", seg_column='seg_path', class_column='class_column', for_mask2former=False):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.seg_column = seg_column
        self.class_column = class_column
        
        self.processor = Mask2FormerImageProcessor(
            do_resize=True,
            size={"height": 1024, "width": 1024},
            ignore_index=255,
            do_normalize=True,
            reduce_labels=False,
        )

        self.df_subject = self.df[self.img_column].drop_duplicates().reset_index()

    def __len__(self):
        return len(self.df_subject)

    def __getitem__(self, idx):
        subject = self.df_subject.iloc[idx][self.img_column]  # frame path
        img_path = os.path.join(self.mount_point, subject)

        df_patches = self.df.loc[self.df[self.img_column] == subject]

        img_t = torch.tensor(np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(img_path)).copy())).to(torch.float32)
        img_t /= 255
        shape = img_t.shape[:2]
        self.current_img = img_path

        bbx = []
        masks = []
        labels = []

        # Create targets for model
        for j, row in df_patches.iterrows():
            label = row[self.class_column]
            labels.append(torch.tensor(label).unsqueeze(0))

            seg_path = os.path.join(self.mount_point, row[self.seg_column])
            seg_t = torch.tensor(np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).copy())).to(torch.float32)


            x, y, w, h = row['x'] * shape[1], row['y'] * shape[0], row['w'] * shape[1], row['h'] * shape[0]
            bb = torch.tensor([np.clip(x, 0, shape[1] - 5),
                            np.clip(y, 0, shape[0] - 5),
                            np.clip((x + w + 1), 5, shape[1]),
                            np.clip((y + h + 1), 5, shape[0])])

            bbx.append(bb.unsqueeze(0))
            masks.append(seg_t.unsqueeze(0))

        masks = torch.cat(masks)
        bbx = torch.cat(bbx, axis=0)
        labels = torch.cat(labels)

        d_augmented = self.transform(image=img_t.numpy(), bboxes=bbx.numpy(), category_ids=labels.numpy(), mask=masks.numpy())
        img_t = d_augmented['image']
        masks = d_augmented['mask']
        boxes = d_augmented['bboxes']


        instance_seg_map = np.zeros(img_t.shape[:2], dtype=np.int32)
        instance_id_to_semantic_id = {}

        for inst_id, (mask, class_id) in enumerate(zip(masks, labels), start=1):
            instance_seg_map[mask > 0] = inst_id
            instance_id_to_semantic_id[inst_id] = int(class_id)


        if len(np.unique(instance_seg_map)) == 1: # only 0 -> no object take new sample
            return self.__getitem__(random.randint(0, len(self) - 1))

        else:
            inputs = self.processor.encode_inputs(
                                    pixel_values_list=[torch.tensor(img_t).permute(2, 0, 1)],
                                    segmentation_maps=[instance_seg_map],
                                    instance_id_to_semantic_id=[instance_id_to_semantic_id],
                                    reduce_labels=False,
                                    ignore_index=0,
                                    return_tensors="pt")
        
            d = inputs
            return d


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

class HystDataModuleSeg(pl.LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", seg_column="seg_path", class_column = 'class_column',balanced=False, train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.class_column = class_column
        self.seg_column = seg_column  
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = monai.data.Dataset(data=HystDatasetSeg(self.df_train, mount_point=self.mount_point, img_column=self.img_column,class_column=self.class_column, seg_column=self.seg_column, transform=self.train_transform))
        self.val_ds = monai.data.Dataset(HystDatasetSeg(self.df_val, mount_point=self.mount_point, img_column=self.img_column,class_column=self.class_column, seg_column=self.seg_column, transform=self.valid_transform))
        self.test_ds = monai.data.Dataset(HystDatasetSeg(self.df_test, mount_point=self.mount_point, img_column=self.img_column,class_column=self.class_column, seg_column=self.seg_column, transform=self.test_transform))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=self.drop_last, collate_fn=self.custom_collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.custom_collate_fn)

    def compute_bb_mask(self, segs, pad=0.1):
        # print(segs.shape)
        shape = segs.shape[1:]
        bbx = []
        for bin_mask in segs:
            ij = torch.argwhere(bin_mask.squeeze() !=0)
            if len(ij) > 1:
                bb = torch.tensor([0, 0, 0, 0])# xmin, ymin, xmax, ymax


                bb[0] = torch.clip(torch.min(ij[:,1]) - shape[1]*pad, 0, shape[1])
                bb[1] = torch.clip(torch.min(ij[:,0]) - shape[0]*pad, 0, shape[0])

                bb[2] = torch.clip(torch.max(ij[:,1]) + shape[1]*pad, 0, shape[1])
                bb[3] = torch.clip(torch.max(ij[:,0]) + shape[0]*pad, 0, shape[0])

            else:
                bb = torch.tensor([0, 0, 1, 1])# xmin, ymin, xmax, ymax

            if bb[0] == bb[2]:
                print(len(ij), bb, ij)
            if bb[1] == bb[3]:
                print(len(ij), bb, ij)

            bbx.append(bb.unsqueeze(0))

        return torch.cat(bbx)


    def custom_collate_fn(self,batch):
        targets = []
        imgs = []
        # pdb.set_trace()
        for sample in batch:
            img = sample.pop('img', None)
            # sample['boxes'] = self.compute_bb_mask(sample['masks'], pad=0.1)
            imgs.append(img.unsqueeze(0))
            targets.append(sample)
        return torch.cat(imgs), targets

class HystDataModuleFormer(pl.LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", seg_column="seg_path", class_column = 'class_column',balanced=False, train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.class_column = class_column
        self.seg_column = seg_column  
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = monai.data.Dataset(HystDatasetFormer(self.df_train, mount_point=self.mount_point, img_column=self.img_column,class_column=self.class_column, seg_column=self.seg_column, transform=self.train_transform))
        self.val_ds = monai.data.Dataset(HystDatasetFormer(self.df_val, mount_point=self.mount_point, img_column=self.img_column,class_column=self.class_column, seg_column=self.seg_column, transform=self.valid_transform))
        self.test_ds = monai.data.Dataset(HystDatasetFormer(self.df_test, mount_point=self.mount_point, img_column=self.img_column,class_column=self.class_column, seg_column=self.seg_column, transform=self.test_transform))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=self.drop_last, collate_fn=self.custom_collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.custom_collate_fn)


    def custom_collate_fn(self,batch):
        pixel_values = torch.stack([example["pixel_values"][0] for example in batch])
        pixel_mask = torch.stack([example["pixel_mask"][0] for example in batch])
        class_labels = [example["class_labels"][0] for example in batch]
        mask_labels = [example["mask_labels"][0] for example in batch]
        return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels}

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


class TrainTransformsSeg:
  def __init__(self):
    # image augmentation functions
    color_jitter = transforms.ColorJitter(brightness=[.5, 1.8], contrast=[0.5, 1.8], saturation=[.5, 1.8], hue=[-.2, .2])
    self.train_transform = Compose(
      [
        # RandZoomd(keys=["img", "masks"], prob=0.5, min_zoom=0.5, max_zoom=1.5, mode=["area", "nearest"], padding_mode='constant'),
        # SquarePad(keys=["img", "masks"]),
        Resized(keys=["img", "masks"], spatial_size=[512, 512], mode=['area', 'nearest']),
        # RandFlipd(keys=["img", "masks"], prob=0.5, spatial_axis=1),
        # RandRotated(keys=["img", "masks"], prob=0.5, range_x=math.pi/2.0, range_y=math.pi/2.0, mode=["bilinear", "nearest"], padding_mode='zeros'),
        ScaleIntensityd(keys=["img"]),
        Lambdad(keys=['img'], func=lambda x: color_jitter(x))
      ]
    )
  def __call__(self, inp):
    return self.train_transform(inp)

class EvalTransformsSeg:
  def __init__(self):
    self.eval_transform = Compose(
      [
        # SquarePad(keys=["img", "masks"]),
        Resized(keys=["img", "masks"], spatial_size=[512, 512], mode=['area', 'nearest']),
        ScaleIntensityd(keys=["img"])                
      ]
    )
  def __call__(self, inp):
    return self.eval_transform(inp)
  
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

class SquarePad:
    def __init__(self, keys):
        self.keys = keys
    def __call__(self, X):

        max_shape = []
        for k in self.keys:
            max_shape.append(torch.max(torch.tensor(X[k].shape)))
        max_shape = torch.max(torch.tensor(max_shape)).item()
        
        return Padd(self.keys, padder=SpatialPad(spatial_size=(max_shape, max_shape)))(X)



class BBXImageTrainTransform():
    def __init__(self):

        self.transform = A.Compose(
            [
                A.LongestMaxSize(max_size_hw=(480, None)),
                A.CenterCrop(height=480, width=836, pad_if_needed=True),
                A.HorizontalFlip(),
                A.GaussNoise(),
                A.OneOf(
                    [
                        A.MotionBlur(p=.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ], p=0.2),
                A.OneOf(
                    [
                        A.OpticalDistortion(p=0.3),
                        A.GridDistortion(p=.1),
                        ], p=0.2),
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=2),
                        A.RandomBrightnessContrast(),
                    ], p=0.3),
                A.HueSaturationValue(p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),

            ], 
            bbox_params=A.BboxParams(format='pascal_voc', min_area=32, min_visibility=0.1, label_fields=['category_ids']),
            additional_targets={'mask': 'masks'}
        )

    def __call__(self, image, bboxes, category_ids, mask):
        return self.transform(image=image, bboxes=bboxes, category_ids=category_ids, mask=mask)

class BBXImageEvalTransform():
    def __init__(self):

        self.transform = A.Compose(
            [
                A.LongestMaxSize(max_size_hw=(480, None)),
                A.CenterCrop(height=480, width=836, pad_if_needed=True),
            ], 
            bbox_params=A.BboxParams(format='pascal_voc', min_area=32, min_visibility=0.1, label_fields=['category_ids']),
            additional_targets={'mask': 'masks'}

        )

    def __call__(self, image, bboxes, category_ids, mask):
        return self.transform(image=image, bboxes=bboxes, category_ids=category_ids, mask=mask)
    

class BBXImageTestTransform():
    def __init__(self):

        self.transform = A.Compose(
            [
                A.LongestMaxSize(max_size_hw=(480, None)),
                A.CenterCrop(height=480, width=836, pad_if_needed=True),
            ], 
            bbox_params=A.BboxParams(format='pascal_voc', min_area=32, min_visibility=0.1, label_fields=['category_ids']),
            additional_targets={'mask': 'masks'}
        )

    def __call__(self, image, bboxes, category_ids, mask):
        return self.transform(image=image, bboxes=bboxes, category_ids=category_ids, mask=mask)


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


