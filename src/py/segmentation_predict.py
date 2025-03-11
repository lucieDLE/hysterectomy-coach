import os 
import pdb
# os.environ['CUDA_VISIBLE_DEVICES']="1"
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
import pandas as pd
import numpy as np 
import pickle
import torch
from tqdm import tqdm 
import SimpleITK as sitk

from nets.segmentation import MaskRCNN
from loaders.hyst_dataset import HystDataModuleSeg, TrainTransformsSeg, EvalTransformsSeg, BBXImageTestTransform

def save_image(filename, np_array):
  writer = sitk.ImageFileWriter()
  writer.SetFileName(filename)
  writer.UseCompressionOn()

  sitk_image = sitk.GetImageFromArray(np_array)
  writer.Execute(sitk_image)
  del sitk_image
  del writer



def construct_multi_mask(bin_masks, labels):
  num_masks = len(labels)
  multi_mask = np.zeros_like(bin_masks[0])
  for i in range(num_masks):
    mask = bin_masks[i]
    label = labels[i]
    multi_mask[mask>0.9] = label
  return multi_mask

def filter_by_confidence(class_ids, scores, masks, boxes, thr):
  """
    Filter detections based on absolute and relative confidence scores. see Notion for example.
    If there are 3 masks with the same labeled, remove the one(s) with lower confidence score(s)

    Args:
        class_ids, scores, masks
        
    Returns:
        Filtered class_ids, scores, and masks
  """

  keep = scores >= thr
  
  class_ids = class_ids[keep]
  scores = scores[keep]
  boxes = boxes[keep]
  masks = masks[keep]
  
  final_keep = []
  unique_classes = np.unique(class_ids)
  
  for class_id in unique_classes:
      class_mask = class_ids == class_id
      if not np.any(class_mask):
          continue
          
      class_scores = scores[class_mask]
      max_score = np.max(class_scores)
      relative_keep = class_scores >= (max_score - (max_score * 0.15))
      
      # Get indices of detections to keep for this class
      keep_idx = np.where(class_mask)[0][relative_keep]
      final_keep.extend(keep_idx)
      
  final_keep = np.array(sorted(final_keep))
    
  return class_ids[final_keep], scores[final_keep], masks[final_keep], boxes[final_keep]

def remove_labels(df, args):
    concats = ['Bipolar', 'Vessel Sealer', 'Robot Grasper Heat', 'Laparoscopic Scissors', 'Laparoscopic Suction', 'Robot Scissors', 'monopolarhook' ]

    if concats is not None:
        replacement_val =20 # df.loc[ df['Instrument Name'] == concats[0]]['class'].unique()
        df.loc[ df['Instrument Name'].isin(concats), "class" ] = replacement_val

    unique_classes = sorted(df[args.class_column].unique())
    class_mapping = {value: idx+1 for idx, value in enumerate(unique_classes)}

    df[args.class_column] = df[args.class_column].map(class_mapping)
    df.rename(columns={concats[0]: 'Others'})

    print(f"{df[[args.class_column]].drop_duplicates()}")
    return df.reset_index()


def main(args):
  
  df_train = pd.read_csv(args.csv_train)
  df_val = pd.read_csv(args.csv_valid)
  df_test = pd.read_csv(args.csv_test)


  df_train = remove_labels(df_train, args)
  df_val = remove_labels(df_val, args)
  df_test = remove_labels(df_test, args)

  print(df_train[ ['Instrument Name', 'class']].value_counts())
  print(df_val[ ['Instrument Name', 'class']].value_counts())
  print(df_test[ ['Instrument Name', 'class']].value_counts())

  num_classes = len(df_train[args.class_column].unique()) + 1 # background

  ttdata = HystDataModuleSeg(df_train, df_val, df_test, batch_size=1, num_workers=args.num_workers, 
                             img_column=args.img_column, seg_column=args.seg_column, class_column=args.class_column, 
                             mount_point=args.mount_point,train_transform=BBXImageTestTransform(),
                             valid_transform=BBXImageTestTransform(), test_transform=BBXImageTestTransform())

  ttdata.setup()
  test_dl = ttdata.test_dataloader()
  test_ds = ttdata.test_ds

  model = MaskRCNN.load_from_checkpoint(args.model)
  model.eval()
  model.cuda()

  outdir_masks = os.path.join(args.out, 'masks/')
  if not os.path.exists(outdir_masks):
    os.makedirs(outdir_masks)

  data_out = []

  # pdb.set_trace()
  with torch.no_grad():
    for idx, batch in tqdm(enumerate(test_dl), total=len(test_dl)): 
      
      img, batch = batch
      outputs = model.forward(img.cuda(), mode='test')

      pred_boxes = outputs[0]['boxes'].cpu().detach().numpy()
      pred_masks = outputs[0]['masks'].cpu().detach().numpy()
      pred_labels = outputs[0]['labels'].cpu().detach().numpy()
      pred_scores = outputs[0]['scores'].cpu().detach().numpy()

      gt_masks = batch[0]['masks']
      gt_boxes = ttdata.compute_bb_mask(gt_masks, pad=0.01).numpy()
      gt_labels = batch[0]['labels'].cpu().detach().numpy()

      # add an else: we should save mask with nothing to compare with ground truth with no detection 
      # -> not really fair to do it like this 
      if (pred_scores >=args.thr_score).any():

        refined_labels, refined_scores, refined_masks, refined_boxes = filter_by_confidence(pred_labels, pred_scores, pred_masks, pred_boxes, thr=args.thr_score)
        
        refined_masks[refined_masks>args.thr_mask] = 1
        refined_masks = refined_masks[:,0,:,:]

        # 2. Build multi - masks
        gt_mul_mask = construct_multi_mask(gt_masks, gt_labels)
        pred_mul_mask = construct_multi_mask(refined_masks, refined_labels)

        # 3. save these masks
        frame_name = test_ds.data.df_subject.iloc[idx][args.img_column].split('img/')[1].split('.nrrd')[0] 

        pred_filename = os.path.join(outdir_masks, frame_name + '_mask_pred.nrrd')
        save_image(pred_filename, pred_mul_mask)

        gt_filename = os.path.join(outdir_masks, frame_name + '_mask_gt.nrrd')
        save_image(gt_filename, gt_mul_mask)

        # 1. save bounding boxes, scores and labels in pickle
        data_out.append({'gt_boxes': gt_boxes,
                        'gt_labels': gt_labels,
                        'gt_mask_path': gt_filename,
                        'pred_boxes': refined_boxes,
                        'pred_labels': refined_labels,
                        'pred_scores': refined_scores,
                        'pred_mask_path': pred_filename,
                        })


  # 4. save labels and scores for analysis
  outfile = os.path.join(args.out, 'predictions.pickle')
  with open(outfile, 'wb') as handle:
    pickle.dump(data_out, handle, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == '__main__':


  parser = argparse.ArgumentParser(description='Surgical Tool Segmentation Training')
  
  input_group = parser.add_argument_group('Input')
  input_group.add_argument('--model', help='Model checkpoint', type=str, required=True)
  input_group.add_argument('--csv_train', help='CSV for training', type=str, required=True)
  input_group.add_argument('--csv_valid', help='CSV for validation', type=str, required=True)
  input_group.add_argument('--csv_test', help='CSV for testing', type=str, required=True)
  input_group.add_argument('--img_column', help='Image/video column name', type=str, default='img_path')
  input_group.add_argument('--seg_column', help='segmentation column name', type=str, default='seg_path')
  input_group.add_argument('--class_column', help='Class column', type=str, default='class_column')
  input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
  input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
  
  input_group.add_argument('--thr_score', help='confidence score threshold to select best predictions', type=float, default=0.2)
  input_group.add_argument('--thr_mask', help='mask threshold to cleanup mask', type=float, default=0.1)
  output_group = parser.add_argument_group('Output')
  output_group.add_argument('--out', help='Output', type=str, default="./")
  
  args = parser.parse_args()
  main(args)