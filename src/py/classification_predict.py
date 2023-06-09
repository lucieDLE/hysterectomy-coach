import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch.utils.data import DataLoader

from nets.classification import HystNet
from loaders.hyst_dataset import HystDataset, TestTransforms
from callbacks.logger import ImageLogger

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from sklearn.utils import class_weight
from sklearn.metrics import classification_report

from tqdm import tqdm

import pickle

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    
    test_fn = args.csv
    
    df_test = pd.read_csv(test_fn)

    
    # ttdata = HystDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, class_column=args.class_column, mount_point=args.mount_point, train_transform=TrainTransforms(), valid_transform=EvalTransforms())

    use_class_column = False
    if args.class_column is not None and args.class_column in df_test.columns:
        use_class_column = True

    model = HystNet(args).load_from_checkpoint(args.model)
    model.eval()
    model.cuda()

    test_ds = HystDataset(df_test, args.mount_point, img_column=args.img_column, class_column=args.class_column, transform=TestTransforms(num_frames=args.num_frames))

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)

    with torch.no_grad():

        predictions = []
        probs = []
        features = []
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for idx, X in pbar: 
            if use_class_column:
                X, Y = X
            X = X.cuda().contiguous()   
            if args.extract_features:        
                pred, x_f = model(X)    
                features.append(x_f.cpu().numpy())
            else:
                pred = model(X)
            probs.append(pred.cpu().numpy())        
            pred = torch.argmax(pred, dim=1).cpu().numpy()
            # pbar.set_description("prediction: {pred}".format(pred=pred))
            predictions.append(pred)
            

    df_test[args.pred_column] = np.concatenate(predictions, axis=0)
    probs = np.concatenate(probs, axis=0)    


    out_dir = os.path.join(args.out, os.path.splitext(os.path.basename(args.model))[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    if use_class_column:
        print(classification_report(df_test[args.class_column], df_test[args.pred_column]))

    ext = os.path.splitext(args.csv)[1]
    if(ext == ".csv"):
        df_test.to_csv(os.path.join(out_dir, os.path.basename(args.csv).replace(".csv", "_prediction.csv")), index=False)
    else:        
        df_test.to_parquet(os.path.join(out_dir, os.path.basename(args.csv).replace(".parquet", "_prediction.parquet")), index=False)

    

    pickle.dump(probs, open(os.path.join(out_dir, os.path.basename(args.csv).replace(ext, "_probs.pickle")), 'wb'))

    if len(features) > 0:
        features = np.concatenate(features, axis=0)
        pickle.dump(features, open(os.path.join(out_dir, os.path.basename(args.csv).replace(ext, "_prediction.pickle")), 'wb'))


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Classification predict')
    parser.add_argument('--csv', type=str, help='CSV file for testing', required=True)
    parser.add_argument('--extract_features', type=int, help='Extract the features', default=0)
    parser.add_argument('--img_column', help='Image/video column name', type=str, default='vid_path')
    parser.add_argument('--class_column', help='Class column', type=str, default='class')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--model', help='Model path to continue training', type=str, default=None)
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--out', help='Output directory', type=str, default="./")
    parser.add_argument('--pred_column', help='Output column name', type=str, default="pred")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=1)
    parser.add_argument('--num_frames', help='Number of frames for the prediction', type=int, default=512)
    parser.add_argument('--nn', help='Type of neural network', type=str, default="efficientnet_b0")

    args = parser.parse_args()

    main(args)
