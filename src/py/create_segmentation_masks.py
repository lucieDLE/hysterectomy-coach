import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # put -1 to not use any

import sys
sys.path.append('/home/lumargot/SurgicalSAM/segment-anything')


import numpy as np
import pandas as pd 
import os 
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import os 
import imageio
import cv2
from PIL import Image
import SimpleITK as sitk
import argparse

def save_image(filename, np_array):
  writer = sitk.ImageFileWriter()
  writer.SetFileName(filename)
  writer.UseCompressionOn()

  sitk_image = sitk.GetImageFromArray(np_array)
  writer.Execute(sitk_image)



def main(args):

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  sam = sam_model_registry["vit_h"](checkpoint=args.sam_model).to(device=device)
  mask_predictor = SamPredictor(sam)

  df = pd.read_csv(args.csv)
  # df = df.loc[df['Instrument Name'] != 'Laparoscopic Scissors'] # 9
  # df = df.loc[df['Instrument Name'] != 'Laparoscopic Suction']  # 10

  df['Instrument Name'].unique()
  unique_classes = sorted(df['Instrument Name'].unique())
  # class_mapping = {value: idx for idx, value in enumerate(unique_classes,1)}


  dict = {'Bipolar':1, 'Laparoscopic Grasper':2, 'Laparoscopic Needle Driver':3,'Needle':4, 'Robot Needle Driver':5, 'Robot Grasper':6, 
          'Vessel Sealer':7, 'Robot Grasper Heat':8, 'Laparoscopic Scissors': 9, 'Laparoscopic Suction':10, 'Robotic Scissors':11}


  for class_name in unique_classes:
    if class_name not in dict.keys():
      dict[class_name] = len(dict.keys()) + 1

  print(dict)

  df["class_column"] = df["Instrument Name"].map(dict)

  # df.to_csv(args.csv)
  print(df[['Instrument Name', 'class_column']].value_counts())
  

  videos = df['Video Name'].unique()

  if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)

  for vid_name in videos:

    video_path = os.path.join(args.mount_point, vid_name)
    cap = cv2.VideoCapture(video_path)
    print(f"is cap opened: {cap.isOpened()}, with {cap.get(cv2.CAP_PROP_FRAME_COUNT)} frames")

    df_vid = df.loc[ df['Video Name'] == vid_name]
  
    frame_indices = df_vid['Frame'].unique()
    
    for frame_idx in frame_indices:
      df_frame = df_vid.loc[ df_vid['Frame'] == frame_idx]
      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
      success, frame = cap.read()
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      
      # img_name = vid_name.split('Cuff_Closure/')[1].replace('.mp4', f"_frame_{frame_idx}.nrrd")
      # print(img_name)
      # filename = os.path.join(args.out_dir, 'img', img_name)
      # save_image(filename, frame)

      # multiple_masks = np.zeros_like(frame[:,:,0])

      for idx, row in df_frame.iterrows():
        frame_idx = row['Frame']
        label = row['Instrument Name']
        x,y,w,h = row['x'], row['y'], row['w'], row['h']
        class_idx = dict[label]

        y1, x1 = y*frame.shape[0], x*frame.shape[1]
        y2, x2 =(y+h)*frame.shape[0], (x+w)*frame.shape[1]

        bbx = np.array([[x1, y1, x2, y2]])

        mask_predictor.set_image(frame)
        masks, scores, logits = mask_predictor.predict(box = bbx, multimask_output=False, point_coords=None, point_labels=None)
        mask = masks[0,...].astype(int)
        # multiple_masks[ mask ==1] = class_idx

        mask_name = vid_name.split('Cuff_Closure/')[1].replace('.mp4', f"_frame_{frame_idx}_class_{class_idx}_n_{idx}.nrrd")
        filename = os.path.join(args.out_dir,'bin_seg', mask_name)  
        save_image(filename, mask)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--sam_model', help='path to sam model', type=str)
    parser.add_argument('--csv', help='csv containing the frames to extract with corresponding boxes and labels', type=str)
    parser.add_argument('--out_dir', help='output directory', type=str, default='./img/')
    parser.add_argument('--mount_point', help='path data dir', type=str, default='./')


    args = parser.parse_args()

    main(args)
