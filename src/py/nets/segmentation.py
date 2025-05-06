from torchvision import models
import torch
import pytorch_lightning as pl
from tqdm import tqdm
import pdb
from torch import nn
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss, maskrcnn_loss, maskrcnn_inference
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import MaskRCNN
from collections import defaultdict
import numpy as np

from utils import FocalLoss
from utils import *
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

class CustomRoIHeads(RoIHeads):
    def __init__(self, *args, hparams=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.hparams = hparams
        self.class_weights = torch.tensor(self.hparams.class_weights, dtype=torch.float32, device='cuda')
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0, weights=self.class_weights)


    def forward(self, features, proposals, image_shapes, targets=None):
        result = []

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result= []
        losses = {}
        if self.training:
            _, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            labels = torch.cat(labels, dim=0)

            loss_classifier = self.ce_loss(class_logits, labels)
            # yhot = nn.functional.one_hot(labels, num_classes=len(self.hparams.class_weights)).float()
            # loss_classifier = self.focal_loss(class_logits, yhot)

            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append({"boxes": boxes[i], "labels": labels[i], "scores": scores[i],})

        mask_proposals = [p["boxes"] for p in result]
        if self.training:
            if matched_idxs is None:
                raise ValueError("if in training, matched_idxs should not be None")

            # during training, only focus on positive boxes
            num_images = len(proposals)
            mask_proposals = []
            pos_matched_idxs = []
            for img_id in range(num_images):
                pos = torch.where(labels[img_id] > 0)[0]
                mask_proposals.append(proposals[img_id][pos])
                pos_matched_idxs.append(matched_idxs[img_id][pos])
        else:
            pos_matched_idxs = None

        if self.mask_roi_pool is not None:
            mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)
        else:
            raise Exception("Expected mask_roi_pool to be not None")

        loss_mask = {}
        if self.training:
            if targets is None or pos_matched_idxs is None or mask_logits is None:
                raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

            gt_masks = [t["masks"] for t in targets]
            gt_labels = [t["labels"] for t in targets]
            rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
            loss_mask = {"loss_mask": rcnn_loss_mask}
        else:
            labels = [r["labels"] for r in result]
            masks_probs = maskrcnn_inference(mask_logits, labels)
            for mask_prob, r in zip(masks_probs, result):
                r["masks"] = mask_prob

        losses.update(loss_mask)

        return result, losses


class MaskRCNN(pl.LightningModule):
  def __init__(self, **kwargs):
    super(MaskRCNN, self).__init__()        
    
    self.save_hyperparameters()
    # anchor_sizes = ((64,), (128,), (256,), (512,), (768,),(1024,),)
    # aspect_ratios = ((0.25, 0.5, 1.0, 2.0, 4.0),) * len(anchor_sizes)

    # anchor_sizes = ((64,), (128,), (256,), (512,), (1024,))
    # aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    # anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    """
    setting num 1: too many False Negative
    rpn_nms_thresh=0.5
    rpn_score_thresh=0.1,
    (roi_head) score_thresh=0.7,
    (roi_head) nms_thresh=0.3,

    setting num 2: less FN but higher FP. Results are better with these
    rpn_nms_thresh=0.6,
    rpn_score_thresh=0.05,
    (roi_head) score_thresh=0.5,
    (roi_head) nms_thresh=0.4,

    
    """

    self.model = models.detection.maskrcnn_resnet50_fpn(weights=models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                                                        rpn_post_nms_top_n_train=500,
                                                        rpn_post_nms_top_n_test=200,
                                                        rpn_nms_thresh=0.8,
                                                        rpn_score_thresh=0.05,
                                                        rpn_fg_iou_thresh=0.7,
                                                        rpn_bg_iou_thresh=0.6,
                                                        rpn_batch_size_per_image=256, #default=256. Try increased and decreased

                                                        )

    # ['loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg'])

    in_features = self.model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    self.model.roi_heads = CustomRoIHeads(box_roi_pool = self.model.roi_heads.box_roi_pool,
                                          box_head =self.model.roi_heads.box_head,
                                          box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, self.hparams.out_features),
                                          
                                          mask_roi_pool=self.model.roi_heads.mask_roi_pool,
                                          mask_head=self.model.roi_heads.mask_head,
                                          mask_predictor=models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, self.hparams.out_features),

                                          fg_iou_thresh=0.4,
                                          bg_iou_thresh=0.3,
                                          batch_size_per_image=512,
                                          bbox_reg_weights=None,
                                          positive_fraction=0.25,
                                          score_thresh=0.3,
                                          nms_thresh=0.5,
                                          detections_per_img=5,
                                          hparams = self.hparams,
                                          )

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
    return optimizer

  def forward(self, images, targets=None, mode='train'):
    if mode == 'train':

      self.model.train()
      losses = self.model(images, targets)
      return losses

    if mode == 'val': # get the boxes and losses
      with torch.no_grad():
        self.model.train()
        losses = self.model(images, targets)
        
        self.model.eval()
        preds = self.model(images)
        self.model.train()
        return [losses, preds]

    elif mode == 'test': # prediction
      self.model.eval()
      output = self.model(images)
      return output

  def training_step(self, train_batch, batch_idx):
      
      imgs, targets = train_batch
      total_loss =0
      loss_dict = self(imgs, targets, mode='train')
      for w, n in zip(self.hparams.loss_weights, loss_dict.keys()):
          loss = loss_dict[n]
          total_loss += w*loss
          self.log(f'train/{n}', loss, sync_dist=True)
      self.log('train_loss', total_loss,sync_dist=True,batch_size=self.hparams.batch_size)
                
      return total_loss
              

  def validation_step(self, val_batch, batch_idx):
      imgs, targets = val_batch      
      loss_dict, preds = self(imgs, targets, mode='val')
      total_loss = 0
      for w, n in zip(self.hparams.loss_weights, loss_dict.keys()):
          loss = loss_dict[n]
          total_loss += w*loss
          self.log(f'val/{n}', loss, sync_dist=True)
  
      self.log('val_loss', total_loss,sync_dist=True,batch_size=self.hparams.batch_size)
  def predict_step(self, images):
      outputs = self(images, mode='test')

      seg_stack = []
      for out in outputs:
          masks = out['masks'].cpu().detach()
          seg = self.compute_segmentation(masks, out['labels'])
          seg_stack.append(seg.unsqueeze(0))
      return torch.cat(seg_stack)
  
  def compute_segmentation(self, masks, labels,thr=0.3):
      ## need a smoothing steps I think, very harsh lines
      labels = labels.cpu().detach().numpy()
      seg_mask = torch.zeros_like(masks[0]) 
      for i in range(len(labels)):
        seg_mask[ masks[i]> thr ] = labels[i]
  
      return seg_mask


class Mask2Former(pl.LightningModule):
    def __init__(self, **kwargs):
        super(Mask2Former, self).__init__()
        
        self.save_hyperparameters()
        
        # Load pretrained Mask2Former model
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-instance",num_labels=self.hparams.out_features, ignore_mismatched_sizes=True)
        
        self.processor = Mask2FormerImageProcessor( do_resize=True, size={"height": 1024, "width": 1024}, ignore_index=255, do_normalize=True, reduce_labels=False,)

    def process_outputs(self, images, outputs):
        original_sizes = [(img.shape[0],img.shape[1]) for img in images]  # example sizes

        result = self.processor.post_process_instance_segmentation(outputs, target_sizes=original_sizes)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        return optimizer

    def forward(self, batch):
        outputs = self.model(pixel_values=batch["pixel_values"],
                             mask_labels=batch["mask_labels"],
                             class_labels=batch["class_labels"],
                             )

        return outputs.loss, outputs

    def training_step(self, batch, batch_idx):
        loss, _ = self(batch)
        self.log('train_loss', loss, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self(batch)
                
        self.log('val_loss', loss, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def predict_step(self, batch, batch_idx):
        # Get model outputs
        loss, out = self(batch)
        
        processed_outputs = self.process_outputs(batch['pixel_values'], out)

        # Apply custom processing for merging overlapping masks of the same class
        final_results = []
        for result in processed_outputs:
            segmentation = result["segmentation"]
            segments_info = result["segments_info"]
            
            # Create a list to store merged segments
            processed_segments = self.merge_same_class_segments(segmentation, segments_info)
            
            # Create final segmentation map
            final_seg = torch.zeros_like(torch.tensor(segmentation))
            for segment in processed_segments:
                mask = segmentation == segment["id"]
                final_seg[mask] = segment["label_id"]
            
            final_results.append(final_seg)
        
        return torch.stack(final_results)

    def merge_same_class_segments(self, segmentation, segments_info, iou_threshold=0.5, adjacency_distance=50):
        """Merge segments of the same class that are overlapping or adjacent"""
        
        # Group segments by class
        class_to_segments = defaultdict(list)
        for segment in segments_info:
            class_to_segments[segment["label_id"]].append(segment)
        
        # Process each class separately
        merged_segments = []
        for class_id, segments in class_to_segments.items():
            # If only one segment for this class, no need to merge
            if len(segments) <= 1:
                merged_segments.extend(segments)
                continue
            
            # Find segments to merge
            groups = []
            processed = set()
            
            for i, seg1 in enumerate(segments):
                if i in processed:
                    continue
                
                # Start a new group
                group = [i]
                processed.add(i)
                
                # Find all segments that should be merged with this one
                for j, seg2 in enumerate(segments):
                    if j in processed or i == j:
                        continue
                    
                    # Calculate masks
                    mask1 = segmentation == seg1["id"]
                    mask2 = segmentation == seg2["id"]
                    
                    # Check if masks overlap or are adjacent
                    overlap = np.logical_and(mask1, mask2).sum()
                    if overlap > 0:
                        # Masks overlap
                        group.append(j)
                        processed.add(j)
                        continue
                    
                    # Check if masks are adjacent
                    # This is a simplified approach - you might want a more sophisticated distance calculation
                    dilated_mask1 = binary_dilation(mask1, iterations=adjacency_distance//2)
                    if np.logical_and(dilated_mask1, mask2).sum() > 0:
                        group.append(j)
                        processed.add(j)
                
                groups.append(group)
            
            # Merge segments in each group
            for group in groups:
                if len(group) == 1:
                    merged_segments.append(segments[group[0]])
                    continue
                
                # Find the segment with highest score or largest area as the primary one
                primary_idx = max(group, key=lambda idx: segments[idx].get("score", 0) * segments[idx].get("area", 1))
                merged_segment = segments[primary_idx].copy()
                
                # Update the area to include all segments in the group
                total_area = sum(segments[idx].get("area", 0) for idx in group)
                merged_segment["area"] = total_area
                
                merged_segments.append(merged_segment)
        
        return merged_segments
