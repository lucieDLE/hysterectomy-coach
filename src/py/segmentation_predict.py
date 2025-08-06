import os 
import pdb
os.environ['CUDA_VISIBLE_DEVICES']="0"
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
import pandas as pd
import numpy as np 
import pickle
import torch
from tqdm import tqdm 
import SimpleITK as sitk

from nets.segmentation import MaskRCNN,FasterRCNN
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

# concats = ['Bipolar', 'Vessel Sealer', 'Robot Grasper Heat', 'Laparoscopic Scissors', 'Laparoscopic Suction', 'Robot Scissors', 'monopolarhook' ]
concats = ['Bipolar', 'Vessel Sealer', 'Scissors', 'Suction', 'Robot Scissors', 'monopolarhook' ]



def construct_class_mapping(df_labels, args):

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
    df = df.loc[df[args.label_column] != 'Needle']

    df.loc[ df[args.label_column].isin(concats), args.label_column ] = 'Others'

    df[args.class_column] = df[args.label_column].map(class_mapping)

    print(f"{df[[args.label_column, args.class_column]].drop_duplicates()}")
    return df.reset_index()


def main(args):
  
  df_train = pd.read_csv(args.csv_train)
  df_val = pd.read_csv(args.csv_valid)
  df_test = pd.read_csv(args.csv_test)


  df_labels = pd.concat([df_train, df_val, df_test])
  class_mapping = construct_class_mapping(df_labels, args)

  df_train = remove_labels(df_train, class_mapping,args)
  df_val = remove_labels(df_val, class_mapping,args)
  df_test = remove_labels(df_test, class_mapping,args)

  print(df_train[ [args.label_column, args.class_column]].value_counts())
  print(df_val[ [args.label_column, args.class_column]].value_counts())
  print(df_test[ [args.label_column, args.class_column]].value_counts())

  num_classes = len(df_train[args.class_column].unique()) + 1 # background
  args_params = vars(args)
  unique_classes = np.sort(np.unique(df_train[args.class_column]))
  args_params['out_features'] = len(unique_classes)+1 # background


  ttdata = HystDataModuleSeg(df_train, df_val, df_test, batch_size=1, num_workers=args.num_workers, 
                             img_column=args.img_column, seg_column=args.seg_column, class_column=args.class_column, 
                             mount_point=args.mount_point,train_transform=BBXImageTestTransform(),
                             valid_transform=BBXImageTestTransform(), test_transform=BBXImageTestTransform())

  ttdata.setup()
  test_dl = ttdata.test_dataloader()
  test_ds = ttdata.test_ds

  if args.nn == 'FasterRCNN':
      model = FasterRCNN.load_from_checkpoint(args.model, **vars(args))
  else:
      model = MaskRCNN.load_from_checkpoint(args.model, **vars(args)) 

  # model = MaskRCNN.load_from_checkpoint(args.model)
  model.eval()
  model.cuda()

  out_dir = os.path.splitext(args.model)[0]
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)


  outdir_masks = os.path.join(out_dir, 'masks/')
  if not os.path.exists(outdir_masks):
    os.makedirs(outdir_masks)

  data_out = []

  # pdb.set_trace()
  with torch.no_grad():
    for idx, batch in tqdm(enumerate(test_dl), total=len(test_dl)): 
      
      img, batch = batch
      outputs = model.forward(img.cuda(), mode='test')

      pred_boxes = outputs[0]['boxes'].cpu().detach().numpy()
      # pred_masks = outputs[0]['masks'].cpu().detach().numpy()
      pred_labels = outputs[0]['labels'].cpu().detach().numpy()
      pred_scores = outputs[0]['scores'].cpu().detach().numpy()

      # gt_masks = batch[0]['masks']
      # gt_boxes = ttdata.compute_bb_mask(gt_masks, pad=0.01).numpy()
      gt_boxes = batch[0]['boxes'].cpu().detach().numpy()
      gt_labels = batch[0]['labels'].cpu().detach().numpy()

      # gt_mul_mask = construct_multi_mask(gt_masks, gt_labels)
      # frame_name = test_ds.data.df_subject.iloc[idx][args.img_column].split('img/')[1].split('.nrrd')[0] 
      # gt_filename = os.path.join(outdir_masks, frame_name + '_mask_gt.nrrd')
      # save_image(gt_filename, gt_mul_mask)

      # add an else: we should save mask with nothing to compare with ground truth with no detection 
      # -> not really fair to do it like this 
      # if (pred_scores >=args.thr_score).any():

      #   refined_labels, refined_scores, refined_masks, refined_boxes = filter_by_confidence(pred_labels, pred_scores, pred_masks, pred_boxes, thr=args.thr_score)
        
      #   refined_masks[refined_masks>args.thr_mask] = 1
      #   refined_masks = refined_masks[:,0,:,:]

      #   # 2. Build multi - masks
      #   pred_mul_mask = construct_multi_mask(refined_masks, refined_labels)


      # pred_filename = os.path.join(outdir_masks, frame_name + '_mask_pred.nrrd')
        # save_image(pred_filename, pred_mul_mask)

        # 1. save bounding boxes, scores and labels in pickle
      data_out.append({'gt_boxes': gt_boxes,
                      'gt_labels': gt_labels,
                      'pred_boxes': pred_boxes,
                      'pred_labels': pred_labels,
                      'pred_scores': pred_scores,
                      })


  # 4. save labels and scores for analysis
  outfile = os.path.join(out_dir, 'predictions.pickle')
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
  input_group.add_argument('--class_column', help='Class column', type=str, default='class')
  input_group.add_argument('--label_column', help='label column', type=str, default='label')
  input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
  input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
  input_group.add_argument('--nn', help='neural network type', type=str, default='MaskRCNN')

  input_group.add_argument('--thr_score', help='confidence score threshold to select best predictions', type=float, default=0.2)
  input_group.add_argument('--thr_mask', help='mask threshold to cleanup mask', type=float, default=0.1)
  
  args = parser.parse_args()
  main(args)