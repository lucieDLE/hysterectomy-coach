import numpy as np
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt 
import itertools
from scipy.optimize import linear_sum_assignment

class FocalLoss(nn.Module):
	def __init__(self, alpha=1, gamma=2, reduction='mean', weights =None):
		super(FocalLoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.reduction = reduction
		self.weights = weights

	def forward(self, inputs, targets):
		BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none', weight=self.weights)
		pt = torch.exp(-BCE_loss)
		F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

		if self.reduction == 'mean':
			return F_loss.mean()
		elif self.reduction == 'sum':
			return F_loss.sum()
		else:
			return F_loss


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix, avg:", np.trace(cm)/len(classes))
  else:
      print('Confusion matrix, without normalization')

  plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.3f' if normalize else 'd'
  thresh = .5 if normalize else np.sum(cm)/4
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

  plt.xlabel('Predicted label')
  plt.ylabel('True label')
  plt.tight_layout()

  return cm

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
    
def dice_coef(groundtruth_mask, pred_mask):
  intersect = np.sum(pred_mask*groundtruth_mask)
  total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
  # dice = np.mean(2*intersect/total_sum)
  return 2*intersect/total_sum

def compute_mask_dice(gt_mask, pred_mask) :
  intersection = np.logical_and(pred_mask, gt_mask).sum()
  union = np.logical_or(pred_mask, gt_mask).sum()
  if union == 0 :
    return 0
  else:
    return (2*intersection) / union

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

def filter_by_confidence(class_ids, scores, boxes, thr):
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
    
  return class_ids[final_keep], scores[final_keep], boxes[final_keep]


def match_boxes(gt_boxes, pred_boxes):
  """Match predicted boxes to ground truth boxes using the Hungarian algorithm."""
  
  if len(gt_boxes) == 0 or len(pred_boxes) == 0:
    return [], []

  # Compute cost matrix (distance between centers)
  cost_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))

  for i, (x1_gt, y1_gt, x2_gt, y2_gt) in enumerate(gt_boxes):
    center_gt = np.array([(x1_gt + x2_gt) / 2, (y1_gt + y2_gt) / 2])
    
    for j, (x1_pred, y1_pred, x2_pred, y2_pred) in enumerate(pred_boxes):
      center_pred = np.array([(x1_pred + x2_pred) / 2, (y1_pred + y2_pred) / 2])
      cost_matrix[i, j] = np.linalg.norm(center_gt - center_pred)  # Euclidean distance

  # Solve assignment problem (minimize distance)
  gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

  return gt_indices, pred_indices


def evaluate_with_fp_fn(gt_boxes, pred_boxes):
  if len(pred_boxes.shape) == 1:
    pred_boxes = torch.tensor(pred_boxes).unsqueeze(0)

  gt_indices, pred_indices = match_boxes(gt_boxes, pred_boxes)

  ious = []
  num_pred =0
  for gt_idx, pred_idx in zip(gt_indices, pred_indices):
    num_pred+=1

    iou = compute_bbx_iou(gt_boxes[gt_idx], pred_boxes[pred_idx])
    ious.append(iou)
  
  # Count false positives (unmatched predictions)
  num_false_positives = len(pred_boxes) - len(pred_indices)

  # Count false negatives (unmatched GT boxes)
  num_false_negatives = len(gt_boxes) - len(gt_indices)
  return num_pred, num_false_positives, num_false_negatives, ious, gt_indices, pred_indices
