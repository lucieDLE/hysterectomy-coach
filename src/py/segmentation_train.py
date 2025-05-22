import os 
# os.environ['CUDA_VISIBLE_DEVICES']="0"
import argparse
import pandas as pd
import numpy as np 
import pdb
import torch

from collections import defaultdict
from nets.segmentation import MaskRCNN, Mask2Former, FasterRCNN
from loaders.hyst_dataset import HystDataModuleSeg, TrainTransformsSeg, EvalTransformsSeg,BBXImageEvalTransform,BBXImageTrainTransform,BBXImageTestTransform, HystDataModuleFormer
from callbacks.logger import ImageLogger, ImageSegLogger, ImageFormerLogger
from utils import *

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning import loggers as pl_loggers

from sklearn.utils import class_weight

laparo = [
       'Hyst_JS_1.30.23.mp4', 
       'Hyst_SurgU_3.21.23d_3069_3157.mp4.mp4',
       'Cuff_Closure/Hyst_MedT_3.20.23r_2965_3187.mp4',
       'Hyst_SurgU_3.21.23b.mp4',
       '!Hyst_MedT_3.21.23d_1219_1482.mp4.mp4',
       'Cuff_Closure/!Hyst_MedT_3.20.23w_1604_1612.mp4'
]

robo = ['Hyst_BB_4.14.23.mp4', 
        'Hyst_BB_4.17.23_3630_3847.mp4.mp4',
       'Cuff_Closure/Hyst_DPM_12.14.22e_3462_4190.mp4',
       'Cuff_Closure/Hyst_BB_4.17.23_3936_4064.mp4',
       'Hyst_BB_1.20.23b.mp4',
       'Cuff_Closure/Hyst_BB_1.20.23a_3525_3753.mp4',
       'Hyst_LHC_4.4.2023c.mp4' ]

# concats = ['Bipolar', 'Vessel Sealer', 'Laparoscopic Scissors', 'Laparoscopic Suction', 'Robot Scissors', 'monopolarhook' ]
concats = ['Bipolar', 'Vessel Sealer', 'Scissors', 'Suction', 'Robot Scissors', 'monopolarhook' ]

import pdb

def construct_class_mapping(args, df_labels):

    # df_labels = df_labels.loc[df_labels['Video Name'].isin(robo)]
    df_labels = df_labels.loc[df_labels[args.label_column] != 'Needle']

    df_labels.loc[ df_labels[args.label_column].isin(concats), args.label_column ] = 'Others'

    unique_classes = sorted(df_labels[args.label_column].unique())
    class_mapping = {value: idx+1 for idx, value in enumerate(unique_classes)}
    return class_mapping

def remove_labels(df, class_mapping, args):
    # concats = ['Bipolar', 'Vessel Sealer', 'Robot Grasper Heat', 'Laparoscopic Scissors', 
    #            'Laparoscopic Suction', 'Robot Scissors', 'monopolarhook' ]
    # concats = ['Bipolar', 'Vessel Sealer', 'Grasper Heat', 'Scissors', 
    #            'Suction', 'monopolarhook' ]
    # df = df.loc[df['Video Name'].isin(robo)]

    df = df.loc[df['to_drop'] == 0]

    df = df.loc[df[args.label_column] != 'Needle']

    df.loc[ df[args.label_column].isin(concats), args.label_column ] = 'Others'

    df[args.class_column] = df[args.label_column].map(class_mapping)

    print(f"{df[[args.label_column, args.class_column]].drop_duplicates()}")
    return df.reset_index()


def main(args):
    train_fn = args.csv_train
    valid_fn = args.csv_valid
    test_fn = args.csv_test
    
    df_train = pd.read_csv(train_fn)
    df_val = pd.read_csv(valid_fn)
    df_test = pd.read_csv(test_fn)


    df_labels = pd.concat([df_train, df_val, df_test])
    class_mapping = construct_class_mapping(args, df_labels)
    df_train = remove_labels(df_train, class_mapping,args)
    df_val = remove_labels(df_val, class_mapping,args)
    df_test = remove_labels(df_test, class_mapping,args)

    print(df_train[ [args.label_column, args.class_column]].value_counts())

    args_params = vars(args)
    unique_classes = np.sort(np.unique(df_train[args.class_column]))
    args_params['out_features'] = len(unique_classes)+1 # background


    args_params['class_weights'] = np.ones(args_params['out_features'])
    if args.balanced_weights:
        unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))

        args_params['class_weights'] = np.concatenate((np.array([0]), unique_class_weights))

    elif args.custom_weights:
        args_params['class_weights'] = np.array(args.custom_weights)

    elif args.balance_dataframe:
        ## new way
        df_train = select_balanced_frames(args,df_train, target_per_class=None, max_deviation=0.01)
        df_val = select_balanced_frames(args,df_val, target_per_class=None, max_deviation=0.01)
        g_train = df_train.groupby(args.class_column)
        df_train = g_train.apply(lambda x: x.sample(g_train.size().min())).reset_index(drop=True).sample(frac=1).reset_index(drop=True)
        unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y= df_train[args.class_column]))
        g_val = df_val.groupby(args.class_column)
        df_val = g_val.apply(lambda x: x.sample(g_val.size().min())).reset_index(drop=True).sample(frac=1).reset_index(drop=True)


    print(f"class weights: {args_params['class_weights']}")

    
    np.random.seed(42)
    torch.random.manual_seed(42)
    torch.cuda.random.manual_seed_all(42)

    ## Mask2former
    if args.nn == 'Mask2Former':

        ttdata = HystDataModuleFormer( df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, 
                                    img_column=args.img_column,seg_column=args.seg_column, class_column=args.class_column, 
                                    mount_point=args.mount_point,train_transform=BBXImageTrainTransform(),valid_transform=BBXImageEvalTransform(), test_transform=BBXImageTestTransform())
        model = Mask2Former(**vars(args))
        image_logger = ImageFormerLogger(max_num_image=12, log_steps=args.log_every_n_steps)

    else:
        # Mask RCNN
        ttdata = HystDataModuleSeg( df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, 
                                    img_column=args.img_column,seg_column=args.seg_column, class_column=args.class_column, 
                                    mount_point=args.mount_point,train_transform=BBXImageTrainTransform(),valid_transform=BBXImageEvalTransform(), test_transform=BBXImageTestTransform())

        if args.nn == 'FasterRCNN':
            if args.model: model = FasterRCNN.load_from_checkpoint(args.model, **vars(args), strict=False)
            else:model = FasterRCNN(**vars(args))
        else:
            if args.model: model = MaskRCNN.load_from_checkpoint(args.model, **vars(args), strict=False) 
            else: model = MaskRCNN(**vars(args))
 
        image_logger = ImageSegLogger(max_num_image=12, log_steps=args.log_every_n_steps)

        # for param in model.model.rpn.parameters():
        #     param.requires_grad = False


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss',
        save_last=True,
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    logger =None
    if args.logger == 'tb':
      logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)
    elif args.logger == 'comet':
      logger = pl_loggers.CometLogger(api_key='jvo9wdLqVzWla60yIWoCd0fX2',
                                      project_name='TMRnet', 
                                      workspace='luciedle',
                                      save_dir="logs/")
    elif args.neptune_tags:
      logger = NeptuneLogger(project='ImageMindAnalytics/surgery-tracking',
                             tags=args.neptune_tags,
                             api_key=os.environ['NEPTUNE_API_TOKEN'],
                             log_model_checkpoints=False)

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[ checkpoint_callback, image_logger],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=True),
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=0.3,
    )

    trainer.fit(model, datamodule=ttdata, ckpt_path=args.model)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Surgical Tool Segmentation Training')
    
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--model', help='Model path to continue training', type=str, default=None)
    input_group.add_argument('--csv_train', help='CSV for training', type=str, required=True)
    input_group.add_argument('--csv_valid', help='CSV for validation', type=str, required=True)
    input_group.add_argument('--csv_test', help='CSV for testing', type=str, required=True)
    input_group.add_argument('--img_column', help='Image/video column name', type=str, default='img_path')
    input_group.add_argument('--seg_column', help='segmentation column name', type=str, default='seg_path')
    input_group.add_argument('--class_column', help='Class column', type=str, default='class')
    input_group.add_argument('--label_column', help='label column', type=str, default='Instrument Name')
    input_group.add_argument('--nn', help='neural network type', type=str, default='MaskRCNN')
    
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--concat_labels', type=str, default=None, nargs='+', help='concat labels in dataframe')

    weight_group = input_group.add_mutually_exclusive_group()
    weight_group.add_argument('--balanced_weights', type=int, default=0, help='Compute weights for balancing the data')
    weight_group.add_argument('--custom_weights', type=float, default=None, nargs='+', help='Custom weights for balancing the data')
    weight_group.add_argument('--balance_dataframe', type=int, default=0, help='balance dataframe')

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--lr', '--learning-rate', default=5e-4, type=float, help='Learning rate')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=500)
    hparams_group.add_argument('--patience', help='Max number of patience steps for EarlyStopping', type=int, default=30)
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=16)
    hparams_group.add_argument('--loss_weights', help='custom loss weights [classifier, box_reg, mask, objectness (rpn), box_reg (rpn)]', type=float, nargs='+', default=[1.0, 1.0, 1.0, 1.0, 1.0])

    logger_group = parser.add_argument_group('Logger')
    logger_group.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=5)    
    logger_group.add_argument('--logger', help='tensorboard, neptune or comet to log experiment', type=str, default=None)
    logger_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="hyst_classification")
    logger_group.add_argument('--neptune_tags', help='neptune tag', type=str, nargs='+', default=None)

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output', type=str, default="./")
    
    args = parser.parse_args()
    main(args)