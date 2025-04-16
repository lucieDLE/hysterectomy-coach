import os 
# os.environ['CUDA_VISIBLE_DEVICES']="0"
import argparse
import pandas as pd
import numpy as np 
import pdb
import torch

from collections import defaultdict
from nets.segmentation import MaskRCNN
from loaders.hyst_dataset import HystDataModuleSeg, TrainTransformsSeg, EvalTransformsSeg,BBXImageEvalTransform,BBXImageTrainTransform,BBXImageTestTransform
from callbacks.logger import ImageLogger, ImageSegLogger


from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning import loggers as pl_loggers

from sklearn.utils import class_weight



def select_balanced_frames(args,df, target_per_class=None, max_deviation=0.2):
    """
        Create a balanced dataframe with similar number of frames per instrument class. The frames selected will 
        have all their instruments listed/counted. Thus, we can't just drop randoms frames or instruments.
        
        target_per_class: number of frames per class, if set to None, take the minimun of frames found per class
        max_deviation: to ensure flexibility, we allow the dataframe to be slightly unbalanced, (i.e 0.2 -> 20%)
    
    """
    # Group by img_path to get unique frames and their class composition
    frames_info = df.groupby(args.img_column).agg({
        args.class_column: list,
        'Frame': 'first'
    }).reset_index()
    
    # Count initial class distribution -> to know how many frames exists per instruments
    initial_class_counts = defaultdict(int)
    for classes in frames_info[args.class_column]:
        # Use set to count unique classes per frame
        for cls in set(classes):
            initial_class_counts[cls] += 1
    
    # attention: target_per_class is the number of frames we want. So even if we are selecting the frame of the 
    # smallest class (i.e. class 3 - 2594 instrument ) it might be a smaller number (we get 2492). This is normal:
    # we can have 2 instruments of class 3 in the same frame. That's why the number of frame is < intrument count.

    if target_per_class is None:
        target_per_class = min(initial_class_counts.values())
    
    target_class_frames = {}
    for cls, count in initial_class_counts.items():
        # take the smallest amount of frames avalaible or how many we want
        target_class_frames[cls] = min(count, target_per_class)

    print("Frame distribution:")
    for cls, count in sorted(initial_class_counts.items()):
        print(f"Class {cls}: {count} frames  (Target: {target_class_frames[cls]})")
    

    # Shuffle frames 
    frames_info = frames_info.sample(frac=1, random_state=42).reset_index(drop=True)
    
    selected_frames = set()
    class_frame_counts = defaultdict(int)
    

    # Need to fill the dataframe by the smallest classes to ensure we use all the frames available
    # Without it if target per class = 106 (class 3), if the instrument is on a frame with other 
    # instrument, they are not going to be selected and class 3 ends up with 65 frames < target_per_class

    sorted_classes = sorted(initial_class_counts.keys(), key=lambda x: initial_class_counts[x])
    for cls in sorted_classes:
        for _, row in frames_info.iterrows():
            # If frame has been selected before, skip
            if row[args.img_column] in selected_frames:
                continue
            
            if cls in row[args.class_column]:
                # Check if adding this frame would exceed the number of frame we want
                would_exceed = False
                for frame_cls in set(row[args.class_column]):
                    if class_frame_counts[frame_cls] >= target_per_class*(1 + max_deviation) :
                        would_exceed = True
                        break
                
                
                if not would_exceed:
                    selected_frames.add(row[args.img_column])
                    for frame_cls in set(row[args.class_column]):
                        class_frame_counts[frame_cls] += 1
            
            # if we have enough frame in the class
            if class_frame_counts[cls] >= target_per_class:
                break
        
    selected_df = df[df[args.img_column].isin(selected_frames)].copy()
    
    print("\nFinal frame distribution:")
    for cls in sorted(initial_class_counts.keys()):
        count = sum(1 for _, row in selected_df.iterrows() if row[args.class_column] == cls)
        target = target_class_frames[cls]
        print(f"Class {cls}: {count} frames (Target: {target})")
    
    print(f"\nTotal frames selected: {len(selected_frames)}")
    
    return selected_df

def remove_labels(df, args):
    concats = ['Bipolar', 'Vessel Sealer', 'Robot Grasper Heat', 'Laparoscopic Scissors', 'Laparoscopic Suction', 'Robot Scissors', 'monopolarhook' ]

    if concats is not None:
        replacement_val =20
        df.loc[ df['Instrument Name'].isin(concats), "class" ] = replacement_val

    unique_classes = sorted(df[args.class_column].unique())
    class_mapping = {value: idx+1 for idx, value in enumerate(unique_classes)}

    df[args.class_column] = df[args.class_column].map(class_mapping)
    df.rename(columns={concats[0]: 'Others'})

    print(f"{df[[args.class_column]].drop_duplicates()}")
    return df.reset_index()


def main(args):
    train_fn = args.csv_train
    valid_fn = args.csv_valid
    test_fn = args.csv_test
    
    df_train = pd.read_csv(train_fn)
    df_val = pd.read_csv(valid_fn)
    df_test = pd.read_csv(test_fn)


    df_train = remove_labels(df_train, args)
    df_val = remove_labels(df_val, args)
    df_test = remove_labels(df_test, args)

    print(df_train[ ['Instrument Name', 'class']].value_counts())

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
        df_train = select_balanced_frames(args,df_train, target_per_class=None, max_deviation=0.1)
        df_val = select_balanced_frames(args,df_val, target_per_class=None, max_deviation=0.1)
        
        unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))
        args_params['class_weights'] = np.concatenate((np.array([0]), unique_class_weights))


    print(f"class weights: {args_params['class_weights']}")

    
    np.random.seed(42)
    torch.random.manual_seed(42)
    torch.cuda.random.manual_seed_all(42)
    
    ttdata = HystDataModuleSeg( df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, 
                                img_column=args.img_column,seg_column=args.seg_column, class_column=args.class_column, 
                                mount_point=args.mount_point,train_transform=BBXImageTrainTransform(),valid_transform=BBXImageEvalTransform(), test_transform=BBXImageTestTransform())

    if args.model:
        model = MaskRCNN.load_from_checkpoint(args.model, **vars(args), strict=False)
    else:
        model = MaskRCNN(**vars(args))

    # Train only the RPN and classification head - 1st stage 
    for param in model.model.rpn.parameters():
        param.requires_grad = True
    for param in model.model.roi_heads.parameters():
        param.requires_grad = True

    for param in model.model.backbone.body.layer1.parameters():
        param.requires_grad = False
    for param in model.model.backbone.body.layer2.parameters():
        param.requires_grad = False
    for param in model.model.backbone.body.layer3.parameters():
        param.requires_grad = False
    for param in model.model.backbone.body.layer4.parameters():
        param.requires_grad = True

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss',
        save_last=True,
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    image_logger = ImageSegLogger(max_num_image=12, log_steps=args.log_every_n_steps)
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
    input_group.add_argument('--class_column', help='Class column', type=str, default='class_column')
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