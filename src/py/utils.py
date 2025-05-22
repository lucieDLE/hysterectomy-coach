import numpy as np
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt 
import itertools
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

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

def convert_to_binary_mask(pred_masks):
  pred_masks[pred_masks>0.1] = 1
  if len(pred_masks.shape) >3:
      pred_masks = pred_masks.squeeze()
  if len(pred_masks.shape) == 2:
      pred_masks = pred_masks.unsqueeze(0)
  return pred_masks

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



def isinBox(box1, box2):
  x11, y11, x12, y12 = box1
  x21, y21, x22, y22 = box2

  if (x11 < x21) and (x12 > x22):
    if (y11 < y21) and (y12 > y22):
      return True
  return False

def select_indices_to_keep(pred_labels, pred_boxes, pred_scores, iou_thr = 0.4):
  """
  Remove predictions if 1) box fully contained in another bigger box, 2) if iou > 0.7, 
  3) if most of the box is contained in a bigger box and have the same labels 
  (#TODO: maybe combined them instead)
  """
  n_pred = pred_labels.shape[0]
  indices_to_remove = []
  for j in range(n_pred):
    if j not in indices_to_remove :
      for i in range(n_pred):
        if i!=j: # skip itself
          if isinBox(pred_boxes[j], pred_boxes[i]):
            indices_to_remove.append(i)
          iou, ratios = compute_bbx_iou(pred_boxes[j], pred_boxes[i], return_containment_ratio=True)

          if (iou > iou_thr):
            if pred_scores[i] > pred_scores[j]:
              indices_to_remove.append(j)
            else :
              indices_to_remove.append(i)
          if ratios[1] - ratios[0] >0.5:
            if pred_labels[i] == pred_labels[j]:
              # maybe for this one we should combine them ?
              indices_to_remove.append(i)

          elif (ratios[0] >0.55) and (ratios[1]>0.55): # 2 overlaps that are very close

            if pred_scores[i] > pred_scores[j]:
              indices_to_remove.append(j)
            else :
              indices_to_remove.append(i)


  indices_to_keep = np.linspace(0,n_pred-1, n_pred, dtype=int)
  indices_to_remove = np.unique(indices_to_remove)
  if len(indices_to_remove) > 0:
    indices_to_keep = np.delete(indices_to_keep, indices_to_remove)
  return indices_to_keep

def remove_small_predictions(boxes, min_width=20, min_height=20):
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    keep_indices = (widths >= min_width) & (heights >= min_height)
    return keep_indices

def apply_indices_selection(input_dic, indices):
  output_dic = { k: v[indices] for k, v in input_dic.items()}
  return output_dic


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

def compute_bbx_iou(gt_box, pred_box, return_containment_ratio=False):
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
  if return_containment_ratio:
    ratio_gt = interArea / gt_boxArea
    ratio_pred = interArea / pred_boxArea
    return iou, np.array([ratio_gt, ratio_pred])
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
    
  return class_ids[final_keep], scores[final_keep], boxes[final_keep],final_keep


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


# def evaluate_with_fp_fn(gt_boxes, pred_boxes):
#   if len(pred_boxes.shape) == 1:
#     pred_boxes = torch.tensor(pred_boxes).unsqueeze(0)

#   gt_indices, pred_indices = match_boxes(gt_boxes, pred_boxes)

#   ious = []
#   num_pred =0
#   for gt_idx, pred_idx in zip(gt_indices, pred_indices):
#     num_pred+=1

#     iou = compute_bbx_iou(gt_boxes[gt_idx], pred_boxes[pred_idx])
#     ious.append(iou)
  
#   # Count false positives (unmatched predictions)
#   num_false_positives = len(pred_boxes) - len(pred_indices)

#   # Count false negatives (unmatched GT boxes)
#   num_false_negatives = len(gt_boxes) - len(gt_indices)
#   return num_pred, num_false_positives, num_false_negatives, ious, gt_indices, pred_indices


def select_balanced_frames(args,df, target_per_class=None, max_deviation=0.2):
    """
        Create a balanced dataframe with similar number of frames per instrument class. The frames selected will 
        have all their instruments listed/counted. Thus, we can't just drop randoms frames or instruments.
        
        target_per_class: number of frames per class, if set to None, take the minimun of frames found per class
        max_deviation: to ensure flexibility, we allow the dataframe to be slightly unbalanced, (i.e 0.2 -> 20%)
    
    """
    # Group by img_path to get unique frames and their class composition
    frames_info = df.groupby(args.img_column).agg({
        args.class_column: list,
        'Frame': 'first'
    }).reset_index()
    
    # Count initial class distribution -> to know how many frames exists per instruments
    initial_class_counts = defaultdict(int)
    for classes in frames_info[args.class_column]:
        # Use set to count unique classes per frame
        for cls in set(classes):
            initial_class_counts[cls] += 1
    
    # attention: target_per_class is the number of frames we want. So even if we are selecting the frame of the 
    # smallest class (i.e. class 3 - 2594 instrument ) it might be a smaller number (we get 2492). This is normal:
    # we can have 2 instruments of class 3 in the same frame. That's why the number of frame is < intrument count.

    if target_per_class is None:
        target_per_class = min(initial_class_counts.values())
    
    target_class_frames = {}
    for cls, count in initial_class_counts.items():
        # take the smallest amount of frames avalaible or how many we want
        target_class_frames[cls] = min(count, target_per_class)

    print("Frame distribution:")
    for cls, count in sorted(initial_class_counts.items()):
        print(f"Class {cls}: {count} frames  (Target: {target_class_frames[cls]})")
    

    # Shuffle frames 
    frames_info = frames_info.sample(frac=1, random_state=42).reset_index(drop=True)
    
    selected_frames = set()
    class_frame_counts = defaultdict(int)
    

    # Need to fill the dataframe by the smallest classes to ensure we use all the frames available
    # Without it if target per class = 106 (class 3), if the instrument is on a frame with other 
    # instrument, they are not going to be selected and class 3 ends up with 65 frames < target_per_class

    sorted_classes = sorted(initial_class_counts.keys(), key=lambda x: initial_class_counts[x])
    for cls in sorted_classes:
        for _, row in frames_info.iterrows():
            # If frame has been selected before, skip
            if row[args.img_column] in selected_frames:
                continue
            
            if cls in row[args.class_column]:
                # Check if adding this frame would exceed the number of frame we want
                would_exceed = False
                for frame_cls in set(row[args.class_column]):
                    if class_frame_counts[frame_cls] >= target_per_class*(1 + max_deviation) :
                        would_exceed = True
                        break
                
                
                if not would_exceed:
                    selected_frames.add(row[args.img_column])
                    for frame_cls in set(row[args.class_column]):
                        class_frame_counts[frame_cls] += 1
            
            # if we have enough frame in the class
            if class_frame_counts[cls] >= target_per_class:
                break
        
    selected_df = df[df[args.img_column].isin(selected_frames)].copy()
    
    print("\nFinal frame distribution:")
    for cls in sorted(initial_class_counts.keys()):
        count = sum(1 for _, row in selected_df.iterrows() if row[args.class_column] == cls)
        target = target_class_frames[cls]
        print(f"Class {cls}: {count} frames (Target: {target})")
    
    print(f"\nTotal frames selected: {len(selected_frames)}")
    
    return selected_df

# # Helper function for binary dilation (for adjacency check)
# def binary_dilation(array, iterations=1):
#     """Simple binary dilation implementation"""
#     import numpy as np
#     dilated = np.copy(array)
#     for _ in range(iterations):
#         padded = np.pad(dilated, 1, mode='constant', constant_values=0)
#         dilated = np.zeros_like(dilated)
#         for i in range(3):
#             for j in range(3):
#                 if i == 1 and j == 1:
#                     continue
#                 dilated |= padded[i:i+dilated.shape[0], j:j+dilated.shape[1]]
#     return dilated

# # Custom merging function for inference
# def merge_overlapping_detections(segmentation, segments_info, iou_threshold=0.5, adjacency_distance=50, score_threshold=0.3):
#     """
#     Merge detections of the same class that are either overlapping significantly or adjacent.
#     adjacency_distance: Maximum pixel distance to consider segments as adjacent
#     score_threshold: Minimum score to keep a detection
    
#     merged_segmentation: Updated segmentation map
#     merged_segments_info: Updated segments information
#     """
#     import numpy as np
#     from collections import defaultdict
    
#     filtered_segments = [seg for seg in segments_info if seg.get("score", 1.0) > score_threshold]
    
#     # Group segments by class
#     class_to_segments = defaultdict(list)
#     for segment in filtered_segments:
#         class_to_segments[segment["label_id"]].append(segment)
    
#     merged_segmentation = np.zeros_like(segmentation)
#     next_id = 1 
    
#     merged_segments_info = []
    
#     for class_id, segments in class_to_segments.items():
#         # If only one segment for this class, no need to merge
#         if len(segments) <= 1:
#             for segment in segments:
#                 mask = segmentation == segment["id"]
#                 merged_segmentation[mask] = next_id
                
#                 new_segment = segment.copy()
#                 new_segment["id"] = next_id
#                 merged_segments_info.append(new_segment)
                
#                 next_id += 1
#             continue
        
#         # For each segment, check if it overlaps with or is adjacent to any other segment
#         segment_masks = []
#         for segment in segments:
#             mask = segmentation == segment["id"]
#             segment_masks.append(mask)
        
#         # Check each pair of segments
#         to_merge = []
#         for i in range(len(segments)):
#             for j in range(i+1, len(segments)):
#                 # Check if masks overlap
#                 overlap = np.logical_and(segment_masks[i], segment_masks[j]).sum()
#                 if overlap > 0:
#                     to_merge.append((i, j))
#                     continue
                
#                 # Check if masks are adjacent
#                 # Dilate first mask and check if it overlaps with second mask
#                 from scipy.ndimage import binary_dilation
#                 dilated_mask = binary_dilation(segment_masks[i], iterations=adjacency_distance//2)
#                 if np.logical_and(dilated_mask, segment_masks[j]).sum() > 0:
#                     to_merge.append((i, j))
        
#         # Group segments to merge using a union-find algorithm
#         parent = list(range(len(segments)))
        
#         def find(x):
#             if parent[x] != x:
#                 parent[x] = find(parent[x])
#             return parent[x]
        
#         def union(x, y):
#             parent[find(x)] = find(y)
        
#         for i, j in to_merge:
#             union(i, j)
        
#         # Create groups of segments to merge
#         groups = defaultdict(list)
#         for i in range(len(segments)):
#             groups[find(i)].append(i)
        
#         # Merge each group
#         for group in groups.values():
#             if len(group) == 1:
#                 # Single segment, no merging needed
#                 segment = segments[group[0]]
#                 mask = segment_masks[group[0]]
#                 merged_segmentation[mask] = next_id
                
#                 # Update segment info
#                 new_segment = segment.copy()
#                 new_segment["id"] = next_id
#                 merged_segments_info.append(new_segment)
#             else:
#                 # Merge multiple segments
#                 merged_mask = np.zeros_like(segment_masks[0], dtype=bool)
#                 for idx in group:
#                     merged_mask = np.logical_or(merged_mask, segment_masks[idx])
                
#                 # Use the segment with highest score as the primary one
#                 primary_idx = max(group, key=lambda idx: segments[idx].get("score", 0))
#                 primary_segment = segments[primary_idx].copy()
                
#                 # Update merged mask in output segmentation
#                 merged_segmentation[merged_mask] = next_id
                
#                 # Update segment info
#                 primary_segment["id"] = next_id
#                 primary_segment["area"] = int(merged_mask.sum())
#                 merged_segments_info.append(primary_segment)
            
#             next_id += 1
    
#     return merged_segmentation, merged_segments_info

# def inference_with_merging(model, image, processor, device="cuda", iou_threshold=0.5, adjacency_distance=50, score_threshold=0.3):
    # """Run inference with Mask2Former and apply custom merging logic"""
    # model.to(device)
    # model.eval()
    
    # inputs = processor(images=image, return_tensors="pt")
    # inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # with torch.no_grad():
    #     outputs = model(**inputs)
    
    # # Post-process outputs
    # results = processor.post_process_panoptic_segmentation(
    #     outputs,
    #     target_sizes=[image.size[::-1]] if hasattr(image, 'size') else [image.shape[:2]],
    #     label_ids_to_fuse=set(),  # Don't fuse any labels
    # )[0]
    
    # # Apply custom merging
    # merged_segmentation, merged_segments_info = merge_overlapping_detections(
    #     results["segmentation"],
    #     results["segments_info"],
    #     iou_threshold=iou_threshold,
    #     adjacency_distance=adjacency_distance,
    #     score_threshold=score_threshold
    # )
    
    # return {
    #     "segmentation": merged_segmentation,
    #     "segments_info": merged_segments_info
    # }