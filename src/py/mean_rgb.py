import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from nets.classification import EfficientnetV2s
from loaders.endo_dataset import EndoDataset, EvalTransformsRGB
from callbacks.logger import ImageLogger

from torch.utils.data import Dataset, DataLoader

from sklearn.utils import class_weight
from tqdm import tqdm

def main(args):
    

    train_fn = os.path.join(args.mount_point, 'CompleteImgNames_Phenotype_notna_images_exist_sample.csv')
    img_column = 'img_path'
    df_train = pd.read_csv(train_fn)

    
    endo_ds = EndoDataset(df_train, mount_point=args.mount_point, img_column=img_column, transform=EvalTransformsRGB(256))
    endo_loader = DataLoader(endo_ds, batch_size=1, num_workers=8, persistent_workers=True, pin_memory=True)

    mean_arr = []
    for x in tqdm(endo_loader, total=len(endo_loader)):
        x = x.cuda()
        mean_arr.append(torch.mean(x.reshape(-1, 3), axis=0).cpu().numpy())

    mean_arr = np.array(mean_arr)
    
    df_train['r'] = mean_arr[:,0]
    df_train['g'] = mean_arr[:,1]
    df_train['b'] = mean_arr[:,2]

    df_train.to_csv("CompleteImgNames_Phenotype_notna_images_exist_sample_rgb.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute mean')

    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    args = parser.parse_args()
    

    main(args)
