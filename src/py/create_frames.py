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
import pdb
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


  mp = '/CMF/data/lumargot/hysterectomy/Instrument_Annotations/Reference_Videos'
  video_dir = '/MEDUSA_STOR/jprieto/surgery_tracking/Cuff_Closure/'
  # mp =  video_dir

  all_dir = '/MEDUSA_STOR/jprieto/surgery_tracking/csv'
  img_dir = '/MEDUSA_STOR/jprieto/surgery_tracking/img/'
  seg_dir = '/MEDUSA_STOR/jprieto/surgery_tracking/bin_seg/'
  

  # df = pd.read_csv(os.path.join(data_dir, csv_name))
  df = pd.read_csv(args.csv)
  df_cat = pd.DataFrame()
  # vid_id = df.iloc[0]['Video Name']
  vid_id = os.path.basename(args.csv).split('_Instrument')[0]

  # unique_classes = sorted(df['Instrument Name'].unique())
  unique_classes =sorted(df["tag"].unique())
  fps = 30
  all_vids = os.listdir(video_dir)


  dict = {'Bipolar':1, 'Laparoscopic Grasper':2, 'Laparoscopic Needle Driver':3,'Needle':4, 
          'Robot Needle Driver':5, 'Robot Grasper':6, 'Vessel Sealer':7, 'Robot Grasper Heat':8, 
          'Laparoscopic Scissors': 9, 'Laparoscopic Suction':10, 'Robot Scissors':11, 'monopolarhook':12}


  for class_name in unique_classes:
    if class_name not in dict.keys():
      dict[class_name] = len(dict.keys()) + 1

  print(dict)

  # df["class_column"] = df["Instrument Name"].map(dict)  
  df["class_column"] = df["tag"].map(dict)  

  if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)

  for video_name in all_vids:
    if vid_id in video_name:

      video = os.path.join(mp, vid_id +'.mp4')
      print(video)

      ### needed if starting from fulll surgery video
      f_start, f_end = os.path.splitext(video_name)[0].split('_')[-2:]
      real_start, real_end = int(f_start)*fps, (int(f_end)-1)*fps
      df_selected = df.loc[ df.loc[:, 'Frame'].between(real_start, real_end), : ]\
      
      # df_selected = df

      cap = cv2.VideoCapture(video)
      frames_idx = df_selected['Frame'].unique()
      print(f" is cap opened: {cap.isOpened()}, with {cap.get(cv2.CAP_PROP_FRAME_COUNT)} frames")

      print(f'number of frames: {len(frames_idx)}')
      for frame_id in frames_idx:
        df_tools = df_selected.loc[ df_selected['Frame']== frame_id]

        path = f"{vid_id}_frame_{int(frame_id)}.nrrd"
        img_path = os.path.join(img_dir, path)

        if not os.path.exists(img_path):
          cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
          success, frame = cap.read()
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
          save_image(img_path, frame)
        else:
          print('exists')
          frame = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).copy()

        for idx, row in df_tools.iterrows():

          if 'x' in row.keys():
            # for csv
            x, y, w, h, name = row[['x','y','w','h','Instrument Name']]
            # for mask 
            x1, y1 = x*frame.shape[1], y*frame.shape[0]
            height, width = h*frame.shape[0], w*frame.shape[1]
            x2, y2 = x1+ width, y1+ height

          else:
            # for mask
            x1, y1, x2, y2, name = row[['XTL','YTL','XBR','YBR','tag']]
            width, height = x2 - x1, y2 - y1
            # for csv
            y, x = y1/frame.shape[0], x1/frame.shape[1]
            h, w = height/frame.shape[0], width/frame.shape[1]

          class_idx = dict[name]
          mask_name = path.replace('.nrrd', f"_class_{class_idx}_n_{idx}.nrrd")
          mask_path = os.path.join(seg_dir,mask_name)

          if not os.path.exists(mask_path):
            bbx = np.array([[x1, y1, x2, y2]])

            mask_predictor.set_image(frame)
            masks, scores, logits = mask_predictor.predict(box = bbx, multimask_output=False, point_coords=None, point_labels=None)
            mask = masks[0,...].astype(int)

            save_image(mask_path, mask)


          df_i = pd.DataFrame(data={'Dataset':'David Labeling',
                                    'Video Name':str(vid_id +'.mp4'),
                                    'Frame':int(frame_id),
                                    'Instrument Name':name,
                                    'x':x,
                                    'y':y,
                                    'w':w,
                                    'h':h,
                                    'img_path':os.path.join('img', path),
                                    'class':class_idx,
                                    'seg_path':os.path.join('bin_seg', mask_name),
                                    },
                                    index=[0],
                                  )
          
          df_cat = pd.concat([df_cat, df_i])

  out = os.path.join(all_dir, vid_id +'_cuff_closure.csv')
  df_cat.to_csv(out)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--sam_model', help='path to sam model', type=str)
    parser.add_argument('--csv', help='csv containing the frames to extract with corresponding boxes and labels', type=str)
    parser.add_argument('--out_dir', help='output directory', type=str, default='./img/')
    parser.add_argument('--mount_point', help='path data dir', type=str, default='./')


    args = parser.parse_args()

    main(args)
