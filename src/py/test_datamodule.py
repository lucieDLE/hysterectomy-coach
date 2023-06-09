import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from nets.classification import HystNet
from loaders.hyst_dataset import HystDataModule, TrainTransforms, EvalTransforms

from sklearn.utils import class_weight

def main(args):

    if(os.path.splitext(args.csv_train)[1] == ".csv"):
        df_train = pd.read_csv(args.csv_train)
        df_val = pd.read_csv(args.csv_valid)
        df_test = pd.read_csv(args.csv_test)
    else:
        df_train = pd.read_parquet(args.csv_train)
        df_val = pd.read_parquet(args.csv_valid)
        df_test = pd.read_parquet(args.csv_test)
    
    unique_classes = np.sort(np.unique(df_train[args.class_column]))
    unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))
    
    
    hystdata = HystDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, class_column=args.class_column, mount_point=args.mount_point, train_transform=TrainTransforms(), valid_transform=EvalTransforms())

    hystdata.setup()

    for idx, batch in enumerate(hystdata.val_dataloader()):
        x, y = batch


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Test data module')
    parser.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    parser.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')
    parser.add_argument('--csv_test', required=True, type=str, help='Test CSV')
    parser.add_argument('--img_column', help='Image/video column name', type=str, default='vid_path')
    parser.add_argument('--class_column', help='Class column', type=str, default='class')
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=1)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=1)


    args = parser.parse_args()

    main(args)
