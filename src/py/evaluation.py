from sklearn.metrics import average_precision_score

import pandas as pd
import numpy as np
import torch
from collections import defaultdict
from utils import *
import torch

import numpy as np

from torchvision.ops import box_iou



def compute_iou_masks(pred_masks, gt_masks):
    ious = torch.zeros((len(pred_masks), len(gt_masks)))
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            intersection = torch.logical_and(pm, gm).sum().float()
            union = torch.logical_or(pm, gm).sum().float()
            ious[i, j] = intersection / union if union > 0 else 0.0
    return ious


def compute_dice(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    """Compute Dice score between two binary masks"""
    pred = pred_mask.bool()
    gt = gt_mask.bool()
    intersection = (pred & gt).sum().float()
    total = pred.sum().float() + gt.sum().float()
    return (2 * intersection / total) if total > 0 else torch.tensor(1.0 if (gt == pred).all() else 0.0)


def evaluate_faster_rcnn(ground_truth, predictions, classes, iou_threshold=0.5):
    y_true = []
    y_pred = []
    total_fp = 0
    total_fn = 0
    total_tp = 0


    stats = defaultdict(list)
    for gt, pred in zip(ground_truth, predictions):
        gt_boxes = torch.tensor(gt['boxes'])
        gt_labels = torch.tensor(gt['labels'])

        pred_boxes = torch.tensor(pred['boxes'])
        pred_labels = torch.tensor(pred['labels'])
        pred_scores = torch.tensor(pred['scores'])

        if len(pred_boxes) == 0:
            # All ground truths are missed (false negatives)
            total_fn += len(gt_boxes)
            continue
        if len(gt_boxes) == 0:
            # All predictions are false positives
            total_fp += len(pred_boxes)
            continue

        box_ious = box_iou(pred_boxes, gt_boxes)
        matched_gt = set()
        matched_pred = set()

        for i, score in enumerate(pred_scores):
            max_box_iou, box_match = torch.max(box_ious[i], dim=0)
            matched = (max_box_iou > iou_threshold) and  (pred_labels[i] == gt_labels[box_match])

            stats['scores'].append(score.item())
            stats['box_iou'].append(max_box_iou.item())

            if max_box_iou > iou_threshold  and box_match.item() not in matched_gt:
                y_true.append(gt_labels[box_match].item())
                y_pred.append(pred_labels[i].item())
                matched_gt.add(box_match.item())
                matched_pred.add(i)
                total_tp += 1
        total_fn += len(gt_boxes) - len(matched_gt)
        total_fp += len(pred_boxes) - len(matched_pred)


    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_box_iou = np.mean(stats['box_iou']) if stats['box_iou'] else 0.0

    # Compute AP per class (macro average)
    ap_per_class = {}
    for cls in classes:
        cls_id = classes.index(cls) +1
        y_true_bin = [1 if y == cls_id else 0 for y in y_true]
        y_pred_bin = [1 if y == cls_id else 0 for y in y_pred]
        if any(y_true_bin):  # Only compute AP if this class is present in GT
            ap = average_precision_score(y_true_bin, y_pred_bin)
        else:
            ap = float('nan')  # or 0.0
        ap_per_class[cls] = ap
    valid_aps = [ap for ap in ap_per_class.values() if not np.isnan(ap)]
    mean_ap = np.mean(valid_aps) if valid_aps else 0.0


    df_pred = pd.DataFrame(data={'gt':y_true, 'pred':y_pred })
    data_stats =  { f'precision@IoU>{iou_threshold}': precision,
                   'recall': recall,
                   'f1_score': f1,
                   'mean_box_iou': mean_box_iou,
                   'false_positives': total_fp,
                   'false_negatives': total_fn,
                   'true_positives': total_tp,
                   'average_precision_per_class': ap_per_class,
                   'mean_average_precision': mean_ap,
                   'num_predictions': len(stats['scores']),
                   'labels': classes,
                    'pr_curve_data': { 'scores': stats['scores'],
                                       'matches': (np.array(y_true) == np.array(y_pred)).astype(int).tolist() }
                    }

    
    return df_pred, data_stats


# don't fit memory
def evaluate_mask_rcnn(ground_truth, predictions, classes, iou_threshold=0.5):
    stats = defaultdict(list)
    y_true, y_pred = [], []
    total_tp, total_fp, total_fn = 0, 0, 0

    for gt, pred in zip(ground_truth, predictions):
        gt_boxes = torch.tensor(gt['boxes'])
        gt_labels = torch.tensor(gt['labels'])
        gt_masks = torch.tensor(gt['masks']).bool()

        pred_boxes = torch.tensor(pred['boxes'])
        pred_labels = torch.tensor(pred['labels'])
        pred_scores = torch.tensor(pred['scores'])
        pred_masks = torch.tensor(pred['masks']).bool()

        # -- box evaluation
        if len(pred_boxes) == 0:
            # All ground truths are missed (false negatives)
            total_fn += len(gt_boxes)
            continue
        elif len(gt_boxes) == 0:
            # All predictions are false positives
            total_fp += len(pred_boxes)
            continue
        else:
            box_ious = box_iou(pred_boxes, gt_boxes)
            matched_gt = set()
            matched_pred = set()

            for i, (pred_label, pred_score) in enumerate(zip(pred_labels,pred_scores)):
                max_iou, gt_idx = torch.max(box_ious[i], dim=0)
                label_match = (pred_label == gt_labels[gt_idx])

                stats['scores'].append(pred_score.item())
                stats['box_iou'].append(max_iou.item())

                if max_iou > iou_threshold and label_match and gt_idx.item() not in matched_gt:
                    # True positive
                    total_tp += 1
                    matched_gt.add(gt_idx.item())
                    matched_pred.add(i)

                    y_true.append(gt_labels[gt_idx].item())
                    y_pred.append(pred_label.item())

                    if pred_masks is not None and gt_masks is not None:
                        mask_iou = compute_iou_masks([pred_masks[i]], [gt_masks[gt_idx]])[0, 0]
                        dice = compute_dice(pred_masks[i], gt_masks[gt_idx])
                        stats['mask_iou'].append(mask_iou.item())
                        stats['dice'].append(dice.item())

        total_fn += len(gt_boxes) - len(matched_gt)
        total_fp += len(pred_boxes) - len(matched_pred)


    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Compute AP per class (macro average)
    ap_per_class = {}
    for cls in classes:
        cls_id = classes.index(cls) +1
        y_true_bin = [1 if y == cls_id else 0 for y in y_true]
        y_pred_bin = [1 if y == cls_id else 0 for y in y_pred]
        if any(y_true_bin):  # Only compute AP if this class is present in GT
            ap = average_precision_score(y_true_bin, y_pred_bin)
        else:
            ap = float('nan')  # or 0.0
        ap_per_class[cls] = ap
    valid_aps = [ap for ap in ap_per_class.values() if not np.isnan(ap)]
    mean_ap = np.mean(valid_aps) if valid_aps else 0.0


    df_pred  = pd.DataFrame(data={'gt': y_true, 'pred': y_pred})
    data_stats = {f'precision@IoU>{iou_threshold}': precision,
                 'recall': recall,
                 'f1_score': f1,
                 'mean_box_iou': np.mean(stats['box_iou']) if stats['box_iou'] else 0.0,
                 'mean_mask_iou': np.mean(stats['mask_iou']) if stats['mask_iou'] else 0.0,
                 'mean_dice': np.mean(stats['dice']) if stats['dice'] else 0.0,
                 'false_positives': total_fp,
                 'false_negatives': total_fn,
                 'true_positives': total_tp,
                 'average_precision_per_class': ap_per_class,
                 'mean_average_precision': mean_ap,
                 'labels': classes,
                #  'pr_curve_data': {'scores': stats['scores'], 
                #                    'matches': (np.array(y_true) == np.array(y_pred)).astype(int).tolist() }
    }
    return df_pred, data_stats


def get_prediction_metrics(gt, pred, stats_dic, iou_threshold=0.5):
    total_tp, total_fp, total_fn = 0, 0, 0
    y_true, y_pred = [], []

    gt_boxes = torch.tensor(gt['boxes'])
    gt_labels = torch.tensor(gt['labels'])
    gt_masks = torch.tensor(gt['masks']).bool()

    pred_boxes = torch.tensor(pred['boxes'])
    pred_labels = torch.tensor(pred['labels'])
    pred_scores = torch.tensor(pred['scores'])
    pred_masks = torch.tensor(pred['masks']).bool()

    # -- box evaluation
    if len(pred_boxes) == 0:
        # All ground truths are missed (false negatives)
        total_fn += len(gt_boxes)
    elif len(gt_boxes) == 0:
        # All predictions are false positives
        total_fp += len(pred_boxes)
    else:
        box_ious = box_iou(pred_boxes, gt_boxes)
        matched_gt = set()
        matched_pred = set()

        for i, (pred_label, pred_score) in enumerate(zip(pred_labels,pred_scores)):
            max_iou, gt_idx = torch.max(box_ious[i], dim=0)
            label_match = (pred_label == gt_labels[gt_idx])

            stats_dic['stats_scores'].append(pred_score.item())
            stats_dic['stats_box_iou'].append(max_iou.item())

            if max_iou > iou_threshold and gt_idx.item() not in matched_gt:
                # True positive
                total_tp += 1
                matched_gt.add(gt_idx.item())
                matched_pred.add(i)

                y_true.append(gt_labels[gt_idx].item())
                y_pred.append(pred_label.item())

                if pred_masks is not None and gt_masks is not None:
                    mask_iou = compute_iou_masks([pred_masks[i]], [gt_masks[gt_idx]])[0, 0]
                    dice = compute_dice(pred_masks[i], gt_masks[gt_idx])
                    stats_dic['stats_mask_iou'].append(mask_iou.item())
                    stats_dic['stats_dice'].append(dice.item())

        total_fn += len(gt_boxes) - len(matched_gt)
        total_fp += len(pred_boxes) - len(matched_pred)

    stats_dic['total_fn'].append(total_fn)
    stats_dic['total_fp'].append(total_fp)
    stats_dic['total_tp'].append(total_tp)

    return  stats_dic, y_true, y_pred

    
def compute_global_metrics(classes, y_true, y_pred, stats, iou_threshold=0.5) :
    tp = sum(stats['total_tp'])
    fp = sum(stats['total_fp'])
    fn = sum(stats['total_fn'])

    mean_box_iou = np.array(stats['stats_box_iou'])
    mean_mask_iou = np.array(stats['stats_mask_iou'])
    mean_dice  = np.array(stats['stats_dice'])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Compute AP per class (macro average)
    ap_per_class = {}
    for cls in classes:
        cls_id = classes.index(cls) +1
        y_true_bin = [1 if y == cls_id else 0 for y in y_true]
        y_pred_bin = [1 if y == cls_id else 0 for y in y_pred]
        if any(y_true_bin):  # Only compute AP if this class is present in GT
            ap = average_precision_score(y_true_bin, y_pred_bin)
        else:
            ap = float('nan')  # or 0.0
        ap_per_class[cls] = ap
    valid_aps = [ap for ap in ap_per_class.values() if not np.isnan(ap)]
    mean_ap = np.mean(valid_aps) if valid_aps else 0.0


    df_pred  = pd.DataFrame(data={'gt': y_true, 'pred': y_pred})
    data_stats = {f'precision@IoU>{iou_threshold}': precision,
                 'recall': recall,
                 'f1_score': f1,
                 'mean_box_iou': np.mean(mean_box_iou) ,
                 'mean_mask_iou': np.mean(mean_mask_iou) ,
                 'mean_dice': np.mean(mean_dice) ,
                 'false_positives': fp,
                 'false_negatives': fn,
                 'true_positives': tp,
                 'average_precision_per_class': ap_per_class,
                 'mean_average_precision': mean_ap,
                 'labels': classes,
                #  'pr_curve_data': {'scores': stats['scores'], 
                #                    'matches': (np.array(y_true) == np.array(y_pred)).astype(int).tolist() }
    }
    return df_pred, data_stats