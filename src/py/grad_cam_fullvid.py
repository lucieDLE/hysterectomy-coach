import argparse

import math
import os
import pandas as pd
import numpy as np 
import pdb
import torch
from torch import nn
from torch.utils.data import DataLoader

# from nets.net_gbp import ResNetLSTM, ResNetLSTM_p2, ResNetLSTM3 ## 
# from nets.classification import ResNetLSTM, ResNetLSTM_p2 # for hysterectomy classification
from nets.end_to_end import ResNetLSTM # for end to end GBP classification


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


def read_and_select_mp4_frames_cv2(vid_path, num_frames):
    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(total_frames):
        success, frame = cap.read()

        frame = cv2.resize(frame, (256,256))
        if success:
            frames.append(frame)
        else:
            frames.append(np.zeros_like(frames[-1]))
        
    video = np.stack(frames, axis=0)

    cap.release()
    cv2.destroyAllWindows()

    return video


def main(args):
    
    test_fn = args.csv
    
    df_test = pd.read_csv(test_fn)

    num_classes = len(np.unique(df_test[args.class_column]))


    scale_intensity = ScaleIntensityRange(0.0, 1.0, 0, 255)

    model = ResNetLSTM.load_from_checkpoint(args.model)

    device = torch.device('cuda')
    model.to(device)
    # model.eval()

    # pdb.set_trace()

    ## hysterectomy or GBP with 2 stages training
    # target_layer = model.model_dict.resnet[-2][-1]

    ## GBP for end-to-end training 
    # pdb.set_trace
    # target_layer = model.model_dict.memory.model_dict.resnet[-2][-1]

    pdb.set_trace()
    # model = model.model_dict.resnet
    target_layer = model.model_dict.resnet[-2]

    target_layers = [target_layer]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers,use_cuda=torch.cuda.is_available())

    # cam = GradCAMPlusPlus(model=model, target_layers=target_layers,use_cuda=torch.cuda.is_available())
    # cam = ScoreCAM(model,target_layers, use_cuda=torch.cuda.is_available())
    
    torch.backends.cudnn.enabled=False

    df_test = df_test.loc[df_test['id'] == 26]
    # pdb.set_trace()


    for target_class in df_test[args.class_column].unique():

        df_class = df_test.loc[df_test[args.class_column]==target_class]
        df_class = df_class.reset_index()

        # if target_class not in [8, 1 ,7,0]:

        for idx, row in df_class.iterrows():

            vid_path = os.path.join(args.mount_point, row[args.img_column])
            vid = read_and_select_mp4_frames_cv2(vid_path, -1)

            vid = torch.from_numpy(vid).float()
            vid_len = vid.shape[0]
            targets = [ClassifierOutputTarget(target_class)]


            out_dir = args.out

            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)

            concat = []
            for idx in range(vid_len-30):

                frame = vid[idx, :, :, :].unsqueeze(0).unsqueeze(0)
                # frame = vid[idx:idx+30, :, :, :].unsqueeze(0)
                frame = frame.transpose(4,2)
                frame = frame.cuda().contiguous()
                frame = scale_intensity(frame)

                # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                gcam_np= cam(input_tensor=frame, eigen_smooth=True)
                # print(gcam_np.min(), gcam_np.max()) ## between either 0 and 0.999 or 0 everywhere
                # concat.append(gcam_np[0].reshape(1,256,-1))
                concat.append(gcam_np)


            concat = np.array(concat)

            out_vid_path = vid_path.replace(vid_path.split('Test')[0], out_dir)

            # print(out_vid_path, vid.shape[1], vid.shape[2])

            out_subdir = os.path.dirname(out_vid_path)

            if not os.path.exists(out_subdir):
                os.makedirs(out_subdir)

            vid_np = vid.squeeze().cpu().numpy().squeeze().astype(np.uint8)
            gcam_np = scale_intensity(concat).squeeze().permute(0,2,1).numpy().astype(np.uint8)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_vid_path, fourcc, args.fps, (vid.shape[2], vid.shape[1]))


            for v, g in zip(vid_np, gcam_np):
                c = cv2.applyColorMap(g, cv2.COLORMAP_JET)
                b = cv2.addWeighted(v, 0.55, c, 0.45, 0)
                out.write(b)


            print(out_vid_path)

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
    output_group.add_argument('--fps', help='Frames per second', type=int, default=1)

    return parser


if __name__ == '__main__':

    parser = get_argparse()
    args = parser.parse_args()

    main(args)