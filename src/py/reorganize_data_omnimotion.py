import os
import re
import shutil
from collections import defaultdict
import SimpleITK as sitk
import numpy as np
import imageio
import pdb
from PIL import Image
import pdb

def save_image(filename, np_array):
  writer = sitk.ImageFileWriter()
  writer.SetFileName(filename)
  writer.UseCompressionOn()

  sitk_image = sitk.GetImageFromArray(np_array)
  writer.Execute(sitk_image)


def convert_nrrd_to_jpeg(nrrd_path, jpeg_path):
    img = sitk.ReadImage(nrrd_path)
    arr = sitk.GetArrayFromImage(img)  # shape: [H, W] or [D, H, W]
    Image.fromarray(arr).save(jpeg_path)


MNT = '/CMF/data/lumargot/hysterectomy/mnt/surgery_tracking/'
# Input folders
IMG_DIR = os.path.join(MNT, 'img')
SEG_DIR = os.path.join(MNT,'bin_seg')
OUTPUT_DIR = os.path.join(MNT,'omnimotion_data/')

# Regex patterns
IMG_PATTERN = re.compile(r'(?P<video>.+)_frame_(?P<frame>\d+)\.nrrd')
SEG_PATTERN = re.compile(r'(?P<video>.+)_frame_(?P<frame>\d+)_class_\d+_n_\d+\.nrrd')

# Organize images by video and frame
# images_by_video = defaultdict(list)
# for fname in os.listdir(IMG_DIR):
#     match = IMG_PATTERN.match(fname)
#     if match:
#         video = match.group('video')
#         frame = int(match.group('frame'))
#         images_by_video[video].append((frame, fname))

ckpt = 0
video_ckpt = 0

video_dir = os.listdir(OUTPUT_DIR)
num_video = len(video_dir)

for video in video_dir:
    print(f"Processing video: {video}")
    video_ckpt +=1

    color_dir = os.path.join(OUTPUT_DIR, video, 'color')
    mask_dir =  os.path.join(OUTPUT_DIR, video, 'mask')

    frame_list = os.listdir(mask_dir)
    frames_per_seq = 156
    total_sequences = len(frame_list) // frames_per_seq

    if len(frame_list) > 0:

        nrrd_path = os.path.join(mask_dir, frame_list[-1])
        jpeg_path = nrrd_path.replace('.nrrd', '.jpeg')
        
        
        img_files = sorted(os.listdir(color_dir))
        mask_files = sorted(os.listdir(mask_dir))

        for i in range(total_sequences):
            print(f'{i}/{total_sequences}')
            seq_name = f'seq{i+1:03d}'
            img_seq_dir = os.path.join(OUTPUT_DIR, video, seq_name, 'color')
            mask_seq_dir = os.path.join(OUTPUT_DIR, video, seq_name, 'mask')

            os.makedirs(img_seq_dir, exist_ok=True)
            os.makedirs(mask_seq_dir, exist_ok=True)

            for j in range(frames_per_seq):
                idx = i * frames_per_seq + j
                img_file = img_files[idx]
                mask_file = mask_files[idx]

                shutil.move(os.path.join(color_dir, img_file), os.path.join(img_seq_dir, img_file))
                shutil.move(os.path.join(mask_dir, mask_file), os.path.join(mask_seq_dir, mask_file))

        print("Finished moving files into sequence folders.")

print("Done.")
