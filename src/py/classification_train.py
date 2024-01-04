import comet_ml
import argparse
import math
import os
import pandas as pd
import numpy as np 

import torch

from nets.classification import HystNet, ResNetLSTM
from loaders.hyst_dataset import HystDataModule, TrainTransforms, EvalTransforms
from callbacks.logger import ImageLogger

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from torchsummary import summary

from sklearn.utils import class_weight

# import neptune
# from neptune import ANONYMOUS_API_TOKEN
from pytorch_lightning.loggers import NeptuneLogger, CometLogger
from pytorch_lightning import loggers as pl_loggers


def main(args):
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    train_fn = args.csv_train
    valid_fn = args.csv_valid
    test_fn = args.csv_test
    
    df_train = pd.read_csv(train_fn)    

    unique_classes = np.sort(np.unique(df_train[args.class_column]))
    unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))    
    df_val = pd.read_csv(valid_fn)            
    
    df_test = pd.read_csv(test_fn)
    
    np.random.seed(42)
    torch.random.manual_seed(42)
    torch.cuda.random.manual_seed_all(42)
    
    ttdata = HystDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, 
                            img_column=args.img_column, class_column=args.class_column, mount_point=args.mount_point, 
                            train_transform=TrainTransforms(num_frames=args.num_frames), 
                            valid_transform=EvalTransforms(num_frames=args.num_frames))


    model = ResNetLSTM(args, out_features=unique_classes.shape[0], class_weights=unique_class_weights)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=30, verbose=True, mode="min")

    image_logger = ImageLogger(num_images=12, log_steps=args.log_every_n_steps)

    if args.logger == 'tb':
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)
    elif args.logger == 'comet':
        logger = pl_loggers.CometLogger(api_key='jvo9wdLqVzWla60yIWoCd0fX2',
                            project_name='TMRnet', 
                            # project_name='TMRnet_2nd_training',
                            workspace='luciedle',
                            save_dir="logs/")

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[ checkpoint_callback, image_logger],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=True),
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
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float, help='Learning rate')
    parser.add_argument('--model', help='Model path to continue training', type=str, default=None)
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=1)    
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=2)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=16)
    parser.add_argument('--num_frames', help='Number of frames', type=int, default=10)
    # parser.add_argument('--nn', help='Type of neural network', type=str, default="efficientnet_v2s")    
    parser.add_argument('--logger', help='tensorboard, neptune or comet to log experiment', type=str, default=None)
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="hyst_classification")

    parser.add_argument('--lfb_model', help='Path to feature bank model', type=str, default=None)
    # parser.add_argument('--valid_lfb', help='Path to validation features', type=str, default=None)
    # parser.add_argument('--lfb_length', help='Number of frames of the memory sequence ', type=int, default=30)


    args = parser.parse_args()

    main(args)
# python3 test_datamodule.py --csv_train /CMF/data/jprieto/hysterectomy/data/Clips/hysterectomy_ds_train_train.csv --csv_valid /CMF/data/jprieto/hysterectomy/data/Clips/hysterectomy_ds_train_test.csv --csv_test /CMF/data/jprieto/hysterectomy/data/Clips/hysterectomy_ds_test.csv 