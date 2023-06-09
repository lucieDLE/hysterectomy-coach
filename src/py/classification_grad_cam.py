import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch import nn
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

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import cv2
from PIL import Image

from monai.transforms import (    
    ScaleIntensityRange
)

class AvgPool1D(nn.Module):
    def __init__(self):
        super(AvgPool1D, self).__init__()
 
    def forward(self, x):
        assert len(x.size()) > 2

        return torch.mean(x, dim=1)


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

    # target_layers = [model.model[0].module.layer4[-1]]

    model = nn.Sequential(
        model.model[0], 
        model.model[1],
        AvgPool1D()
        )
    model.eval()
    model.cuda()

    target_layers = [model[0].module.layer4[-1]]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #   ...

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.

    test_ds = HystDataset(df_test, args.mount_point, img_column=args.img_column, class_column=args.class_column, transform=TestTransforms(num_frames=args.num_frames))

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)

    scale_intensity = ScaleIntensityRange(0.0, 1.0, 0, 255)

    predictions = []
    probs = []
    features = []
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for idx, X in pbar: 
        if use_class_column:
            X, Y = X
        X = X.cuda().contiguous()   
        
        targets = [ClassifierOutputTarget(args.target_class)]

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        gcam_np = cam(input_tensor=X, targets=None)

        # print(np.min(gcam_np), np.max(gcam_np)) -> between 0, 0.9999

        vid_path = df_test.loc[idx][args.img_column]

        out_vid_path = vid_path.replace(os.path.splitext(vid_path)[1], '.avi')

        out_vid_path = os.path.join(args.out, out_vid_path)

        out_dir = os.path.dirname(out_vid_path)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        vid_np = scale_intensity(X).permute(0,1,3,4,2).squeeze().cpu().numpy().squeeze().astype(np.uint8)
        gcam_np = scale_intensity(gcam_np).squeeze().numpy().astype(np.uint8)

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(out_vid_path, fourcc, args.fps, (256, 256))

        for v, g in zip(vid_np, gcam_np):
            c = cv2.applyColorMap(g, cv2.COLORMAP_JET)
            b = cv2.addWeighted(v, 0.5, c, 0.5, 0)
            out.write(b)

        out.release()

        # #batch,time,C,W,H -> batch,time,W,H,C
        # img = sitk.GetImageFromArray(, isVector=True)
        # sitk.WriteImage(img, out_img)

        # grayscale_cam = sitk.GetImageFromArray(grayscale_cam)
        # out_grad = out_img.replace('.nrrd', '_gradcam.nrrd')
        # sitk.WriteImage(grayscale_cam, out_grad)


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
    # parser.add_argument('--nn', help='Type of neural network', type=str, default="efficientnet_b0")
    parser.add_argument('--num_frames', help='Number of frames for the prediction', type=int, default=512)
    parser.add_argument('--target_class', help='Target class', type=int, default=0)
    parser.add_argument('--fps', help='Frames per second', type=int, default=20)


    args = parser.parse_args()

    main(args)
