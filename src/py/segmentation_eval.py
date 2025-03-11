import os 
os.environ['CUDA_VISIBLE_DEVICES']="0"
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


import SimpleITK as sitk
from sklearn.metrics import precision_recall_curve, auc
from nets.segmentation import MaskRCNN
from loaders.hyst_dataset import HystDataModuleSeg, TrainTransformsSeg, EvalTransformsSeg


def filter_by_confidence(class_ids, scores, masks, boxes, thr):
  """
    Filter detections based on absolute and relative confidence scores. see Notion for example.
    If there are 3 masks with the same labeled, remove the one(s) with lower confidence score(s)

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

def compute_mask_iou(gt_mask, pred_mask) :
    """
    Compute IoU between two masks.
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0 :
      return 0
    else:
      return intersection / union

def compute_bbx_iou(gt_box, pred_box):
    """
    Compute IoU for two bounding boxes.
    Each box should be [x1, y1, x2, y2]
    """
    x1 = max(gt_box[0], pred_box[0])
    y1 = max(gt_box[1], pred_box[1])
  
    x2 = min(gt_box[2], pred_box[2])
    y2 = min(gt_box[3], pred_box[3])

    interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    gt_boxArea = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
    pred_boxArea = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    iou = interArea / float(gt_boxArea + pred_boxArea - interArea)
    return iou

def compute_bbx_ap_at_iou(gt_boxes, pred_boxes, pred_scores, iou_threshold):
    """Compute AP for a single IoU threshold."""
    matched = set()
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    
    for i, pred_box in enumerate(pred_boxes):
        if pred_scores[i] < 0.0:  # Skip invalid detections
            fp[i] = 1
            continue
        
        best_iou = 0.0
        best_match = -1
        for j, gt_box in enumerate(gt_boxes):
            if j in matched:
                continue
            iou = compute_bbx_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_match = j

        if best_iou >= iou_threshold:
            tp[i] = 1
            matched.add(best_match)
            
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls = tp_cumsum / len(gt_boxes)

    if len(recalls) < 2 or len(precisions) < 2:
        return 0.0  # Default AP value if AUC cannot be computed

    return auc(recalls, precisions)


def compute_map_one_class(gt_boxes, pred_boxes, pred_scores, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """Compute mAP averaged over multiple IoU thresholds."""
    aps = []
    for iou_threshold in iou_thresholds:
        ap = compute_bbx_ap_at_iou(gt_boxes, pred_boxes, pred_scores, iou_threshold)
        aps.append(ap)
    return np.mean(aps), aps


def compute_map_multiclass(ground_truths, predictions, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Compute mAP for multiclass detection.
    
    Parameters:
        predictions: List of dicts. Each dict contains 'boxes', 'scores', 'labels' for predictions.
        ground_truths: List of dicts. Each dict contains 'boxes', 'labels' for ground truth.
        iou_thresholds: List of IoU thresholds.
    
    Returns:
        mAP: Mean Average Precision across all classes and IoU thresholds.
        class_aps: AP per class.
    """
    all_classes = set()
    for gt in ground_truths:
        for label in gt['labels']:
            all_classes.add(label)

    print(all_classes)
    class_aps = {}

    for cls in all_classes:
        cls_pred_boxes = []
        cls_pred_scores = []
        cls_gt_boxes = []

        # Collect predictions and ground truths for the current class
        for pred, gt in zip(predictions, ground_truths):

            # print((pred), gt) # dict, dict 
            for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
                if label == cls:
                    cls_pred_boxes.append(box)
                    cls_pred_scores.append(score)
            for box, label in zip(gt['boxes'], gt['labels']):
                if label == cls:
                    cls_gt_boxes.append(box)

        if len(cls_gt_boxes) == 0:
            continue  # Skip classes without ground truths

        aps = []
        for iou_threshold in iou_thresholds:
            ap = compute_bbx_ap_at_iou(cls_gt_boxes, cls_pred_boxes, cls_pred_scores, iou_threshold)
            aps.append(ap)
        
        class_aps[cls] = np.mean(aps)

    mAP = np.mean(list(class_aps.values()))
    return mAP, class_aps



df_train = pd.read_csv('/MEDUSA_STOR/jprieto/surgery_tracking/csv/dataset_6_classes_train_train.csv')
df_val = pd.read_csv('/MEDUSA_STOR/jprieto/surgery_tracking/csv/dataset_6_classes_train_test.csv')
df_test = pd.read_csv('/MEDUSA_STOR/jprieto/surgery_tracking/csv/dataset_6_classes_test.csv')


img_column = 'img_path'
seg_column = 'seg_path'
class_column = 'class'
mount_point = '/MEDUSA_STOR/jprieto/surgery_tracking/'
# ckp = os.path.join(mount_point, 'output/test-model', 'epoch=5-val_loss=0.34.ckpt')
# ckpt = '/MEDUSA_STOR/jprieto/surgery_tracking/output/test-6class-cat/epoch=22-val_loss=0.73.ckpt'
ckpt='/MEDUSA_STOR/jprieto/surgery_tracking/output/no_augm/epoch=2-val_loss=0.98.ckpt'


num_classes = len(df_train[class_column].unique()) + 1 # background

g_train = df_train.groupby(class_column)
df_train = g_train.apply(lambda x: x.sample(g_train.size().min())).reset_index(drop=True).sample(frac=1).reset_index(drop=True)

g_val = df_val.groupby(class_column)
df_val = g_val.apply(lambda x: x.sample(g_val.size().min())).reset_index(drop=True).sample(frac=1).reset_index(drop=True)


ttdata = HystDataModuleSeg( df_train, df_val, df_test, batch_size=1, num_workers=4, 
                            img_column=img_column,seg_column=seg_column, class_column=class_column, 
                            mount_point=mount_point,train_transform=TrainTransformsSeg(),
                            valid_transform=EvalTransformsSeg())

ttdata.setup()

test_dl = ttdata.test_dataloader()
test_ds = ttdata.test_ds
model = MaskRCNN.load_from_checkpoint(ckpt)
    
model.eval()
model.cuda()

IOU_THR = 0.6## thr for all mAP



all_predictions, all_ground_truths = [], []
batch_size=4
with torch.no_grad():
    for idx, batch in tqdm(enumerate(test_dl), total=len(test_dl)): 
    
        imgs = []
        img = batch.pop('img', None)

        imgs.append(img)
        imgs = torch.cat(imgs)

        outputs = model.forward(imgs.cuda(), mode='test')
        SCORE_THR = 0.4

        pred_boxes = outputs[0]['boxes'].cpu().detach().numpy()
        pred_masks = outputs[0]['masks'].cpu().detach().numpy()
        pred_labels = outputs[0]['labels'].cpu().detach().numpy()
        pred_scores = outputs[0]['scores'].cpu().detach().numpy()

        gt_masks = batch['masks'][0]
        gt_boxes = ttdata.compute_bb_mask(gt_masks, pad=0.01).numpy()
        gt_labels = batch['labels'][0].cpu().detach().numpy()


        if (pred_scores >=SCORE_THR).any():
            refined_labels, refined_scores, refined_masks, refined_boxes = filter_by_confidence(pred_labels, pred_scores, pred_masks, pred_boxes, thr=SCORE_THR)
            all_predictions.append({
                                    # 'masks': refined_masks,
                                    'boxes': refined_boxes,
                                    'labels': refined_labels,
                                    'scores': refined_scores,
                                    })

            all_ground_truths.append({
                                        # 'masks': gt_masks.numpy(),
                                        'boxes': gt_boxes,
                                        'labels': gt_labels,
                                        })

mAP, class_aps = compute_map_multiclass(all_ground_truths, all_predictions, iou_thresholds=np.arange(0.5, 1.0, 0.05))
print(f"AP: {mAP}")
print(f"class AP\n {class_aps}")

print()
print(len(all_ground_truths))