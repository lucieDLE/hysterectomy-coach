import argparse

import math
import os
import pandas as pd
import numpy as np 
import pdb
import torch
from torch import nn
from torch.utils.data import DataLoader

# from nets.net_gbp import ResNetLSTM, ResNetLSTM_p2, ResNetLSTM3
from nets.classification import ResNetLSTM, ResNetLSTM_p2
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
from torchvision.models import resnet50, ResNet50_Weights

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
    print(df_test[args.class_column].unique())

    for target_class in df_test[args.class_column].unique():

        if target_class == 1 :

            df_class = df_test.loc[df_test[args.class_column]==target_class]
            df_class = df_class.reset_index()
            print(df_class['class'].unique())

            test_ds = HystDataset(df_class, args.mount_point, img_column=args.img_column, class_column=args.class_column,num_frames=args.num_frames, transform=TestTransforms(num_frames=args.num_frames))
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)

            num_classes = len(np.unique(df_test[args.class_column]))

            use_class_column = False
            if args.class_column is not None and args.class_column in df_test.columns:
                use_class_column = True


            # model = ResNetLSTM3.load_from_checkpoint(args.model, strict=True)
                

            if args.eval_bank:
                model = ResNetLSTM(args, out_features=num_classes)
            else:
                model = ResNetLSTM_p2(args, out_features=num_classes)
            
            device = torch.device('cuda')
            model.to(device)
            model.eval()
            
            # 
            # target_layer = model.model[-2]
            # target_layer = model.memory_model.model[-2]
            target_layer = model.model_dict.resnet[-2]


            target_layers = [target_layer]

            # Construct the CAM object once, and then re-use it on many images:
            cam = GradCAM(model=model, target_layers=target_layers)
            torch.backends.cudnn.enabled=False


            targets = None
            if not target_class is None:
                targets = [ClassifierOutputTarget(target_class)]


            scale_intensity = ScaleIntensityRange(0.0, 1.0, 0, 255)
            
            out_dir = args.out

            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)

            pbar = tqdm(enumerate(test_loader), total=len(test_loader))

            # pdb.set_trace()
            for idx, X in pbar:
                if use_class_column:
                    X, Y = X
                X = X.cuda().contiguous()

                concat = []
                for i in range(args.num_frames):
                    Xi = X[:,i,:,:,:]
                    Xi = Xi.reshape((1,1,3,256,256))

                    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                    gcam_np= cam(input_tensor=Xi, targets=targets)
                    concat.append(gcam_np)

                    # print(np.min(gcam_np), np.max(gcam_np)) -> between 0, 0.9999


                concat = np.array(concat)
                concat = concat.reshape((args.num_frames, 1, 256, 256))

                vid_path = df_class.loc[idx][args.img_column]

                out_vid_path = vid_path.replace(os.path.splitext(vid_path)[1], '.mp4')

                out_vid_path = os.path.join(out_dir, out_vid_path)

                out_subdir = os.path.dirname(out_vid_path)

                if not os.path.exists(out_subdir):
                    os.makedirs(out_subdir)

                vid_np = scale_intensity(X).permute(0,1,3,4,2).squeeze().cpu().numpy().squeeze().astype(np.uint8)
                gcam_np = scale_intensity(concat).squeeze().numpy().astype(np.uint8)


                print(vid_np.shape, gcam_np.shape)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(out_vid_path, fourcc, args.fps, (256, 256))

                for v, g in zip(vid_np, gcam_np):
                    c = cv2.applyColorMap(g, cv2.COLORMAP_JET)
                    b = cv2.addWeighted(v, 0.5, c, 0.5, 0)
                    out.write(b)

                out.release()


def get_argparse():

    parser = argparse.ArgumentParser(description='Classification GradCam')

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--csv', type=str, help='CSV file for testing', required=True)
    input_group.add_argument('--img_column', help='Image/video column name', type=str, default='vid_path')
    input_group.add_argument('--class_column', help='Class column', type=str, default='class')
    input_group.add_argument('--num_frames', help='Number of frames for the prediction', type=int, default=512)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    input_group.add_argument('--eval_bank', help='bool to evaluate memory bank or full network', type=bool, default=False)
    input_group.add_argument('--lfb_model', help='path to memory bank', type=str, default=False)


    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--model', help='Model path to continue training', type=str, default=None)
    model_group.add_argument('--target_layer', help='Target layer for GradCam. For example in ResNet, the target layer is the last conv layer which is layer4', type=str, default='layer4')
    model_group.add_argument('--target_class', help='Target class', type=int, default=0)

    model_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    model_group.add_argument('--batch_size', help='Batch size', type=int, default=16)

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    output_group.add_argument('--fps', help='Frames per second', type=int, default=25)

    return parser


if __name__ == '__main__':

    parser = get_argparse()
    args = parser.parse_args()

    main(args)