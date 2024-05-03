import comet_ml
import argparse
import math
import os
import pandas as pd
import numpy as np 
import pdb
import torch

# from nets.classification import HystNet, ResNetLSTM, ResNetLSTM_p2
from nets.end_to_end import *
from loaders.hyst_dataset import HystDataModule, TrainTransforms, EvalTransforms
from callbacks.logger import ImageLogger

from pytorch_lightning import Trainer


from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
# from torchsummary import summary

from sklearn.utils import class_weight

# import neptune
# from neptune import ANONYMOUS_API_TOKEN
from pytorch_lightning.loggers import NeptuneLogger, CometLogger
from pytorch_lightning import loggers as pl_loggers


def main(args):
    
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    train_fn = args.csv_train
    valid_fn = args.csv_valid
    test_fn = args.csv_test
    
    df_train = pd.read_csv(train_fn)
    df_val = pd.read_csv(valid_fn)
    df_test = pd.read_csv(test_fn)

    unique_classes = np.sort(np.unique(df_train[args.class_column]))
    unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))    
    # unique_class_weights = None

    
    np.random.seed(42)
    torch.random.manual_seed(42)
    torch.cuda.random.manual_seed_all(42)
    
    ttdata = HystDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, 
                            img_column=args.img_column, class_column=args.class_column, mount_point=args.mount_point, num_frames=args.num_frames,
                            train_transform=TrainTransforms(num_frames=args.num_frames),
                            valid_transform=EvalTransforms(num_frames=args.num_frames))

    # if args.lfb_model:
    #     model = ResNetLSTM_p2(args, out_features=unique_classes.shape[0], class_weights=unique_class_weights)
    # else:
    #     model = ResNetLSTM(args, out_features=unique_classes.shape[0], class_weights=unique_class_weights)


    model = ResNetLSTM(args, out_features=unique_classes.shape[0], class_weights=unique_class_weights)


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=20, verbose=True, mode="min")

    image_logger = ImageLogger(num_images=12, log_steps=args.log_every_n_steps)

    if args.logger == 'tb':
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)
    elif args.logger == 'comet':
        logger = pl_loggers.CometLogger(api_key='jvo9wdLqVzWla60yIWoCd0fX2',
                            project_name='TMRnet', 
                            workspace='luciedle',
                            experiment_key="72cda6cc6996438f89701727893ef5da",
                            save_dir="logs/")

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[ checkpoint_callback, early_stop_callback, image_logger],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=True),
        log_every_n_steps=args.log_every_n_steps,
        # replace_sampler_ddp=False ## pytorch theia
        use_distributed_sampler = False ## pytorch grond
    )

    trainer.fit(model, datamodule=ttdata, ckpt_path=args.model)

    # pdb.set_trace()

    # ckpt_path = os.path.join(args.out, )
    # torch.save(model.state_dict(), )
    # the_model = ResNetLSTM_p2()
    # for name, param in the_model.named_parameters(): print(f"{name}: {param.data}")

    # pdb.set_trace()

    # model = ResNetLSTM_p2.load_from_checkpoint(args.model, strict=True)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Endometrioid classification Training')
    parser.add_argument('--csv_train', help='CSV for training', type=str, required=True)
    parser.add_argument('--csv_valid', help='CSV for validation', type=str, required=True)
    parser.add_argument('--csv_test', help='CSV for testing', type=str, required=True)
    parser.add_argument('--img_column', help='Image/video column name', type=str, default='vid_path')
    parser.add_argument('--class_column', help='Class column', type=str, default='class')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float, help='Learning rate')
    parser.add_argument('--model', help='Model path to continue training', type=str, default=None)
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=500)
    parser.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=1)    
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
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