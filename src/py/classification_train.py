import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from nets.classification import HystNet
from loaders.hyst_dataset import HystDataModule, TrainTransforms, EvalTransforms
from callbacks.logger import ImageLogger

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from sklearn.utils import class_weight

def main(args):
    

    train_fn = args.csv_train
    valid_fn = args.csv_valid
    test_fn = args.csv_test
    
    df_train = pd.read_csv(train_fn)    

    unique_classes = np.sort(np.unique(df_train[args.class_column]))
    unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))    

    df_val = pd.read_csv(valid_fn)            
    
    df_test = pd.read_csv(test_fn)

    
    ttdata = HystDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, class_column=args.class_column, mount_point=args.mount_point, train_transform=TrainTransforms(num_frames=args.num_frames), valid_transform=EvalTransforms(num_frames=args.num_frames))

    model = HystNet(args, out_features=unique_classes.shape[0], class_weights=unique_class_weights)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=30, verbose=True, mode="min")

    image_logger = ImageLogger(num_images=12, log_steps=args.log_every_n_steps)

    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)    

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, checkpoint_callback, image_logger],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=False),
        log_every_n_steps=args.log_every_n_steps
    )
    trainer.fit(model, datamodule=ttdata, ckpt_path=args.model)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Endometrioid classification Training')
    parser.add_argument('--csv_train', help='CSV for training', type=str, required=True)
    parser.add_argument('--csv_valid', help='CSV for validation', type=str, required=True)
    parser.add_argument('--csv_test', help='CSV for testing', type=str, required=True)
    parser.add_argument('--img_column', help='Image/video column name', type=str, default='vid_path')
    parser.add_argument('--class_column', help='Class column', type=str, default='class')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--model', help='Model path to continue training', type=str, default=None)
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=50)    
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=128)
    parser.add_argument('--num_frames', help='Number of frames', type=int, default=100)
    # parser.add_argument('--nn', help='Type of neural network', type=str, default="efficientnet_v2s")    
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="hyst_classification")


    args = parser.parse_args()

    main(args)
