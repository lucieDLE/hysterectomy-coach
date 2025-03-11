from torchvision import models
import torch
import pytorch_lightning as pl
from tqdm import tqdm
import pdb
from torch import nn
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss, maskrcnn_loss, maskrcnn_inference

class CustomRoIHeads(RoIHeads):
    def __init__(self, *args, hparams=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.hparams = hparams
        self.class_weights = torch.tensor(self.hparams.class_weights, dtype=torch.float32, device='cuda')
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)

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
    # pdb.set_trace()
    
    self.model = models.detection.maskrcnn_resnet50_fpn(weights=models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                                                        rpn_post_nms_top_n_train=150,
                                                        rpn_post_nms_top_n_test=100,
                                                        rpn_nms_thresh=0.8,
                                                        rpn_fg_iou_thresh=0.5,
                                                        rpn_batch_size_per_image=256, #default=256. Try increased and decreased
                                                        detections_per_img=8,

                                                        box_batch_size_per_image=512, #default=512. Increase if most proposal are background. Balanced pos/neg samples
                                                        )

    # ['loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg'])
    self.loss_weights = [1.0 , 1.0, 1, 1.0, 1.0]
    # self.loss_weights = [5, 2, 1, 3.0e-5, 4.0e-5]

    in_features = self.model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    self.model.roi_heads = CustomRoIHeads(box_roi_pool = self.model.roi_heads.box_roi_pool,
                                          box_head =self.model.roi_heads.box_head,
                                          box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, self.hparams.out_features),
                                          
                                          mask_roi_pool=self.model.roi_heads.mask_roi_pool,
                                          mask_head=self.model.roi_heads.mask_head,
                                          mask_predictor=models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, self.hparams.out_features),

                                          fg_iou_thresh=0.5,
                                          bg_iou_thresh=0.5,
                                          batch_size_per_image=512,
                                          bbox_reg_weights=None,
                                          positive_fraction=0.25,
                                          score_thresh=0.5,
                                          nms_thresh=0.3,
                                          detections_per_img=8,
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
      
      loss_dict = self(imgs, targets, mode='train')
      loss = sum([loss for loss in loss_dict.values()])
      self.log('train_loss', loss,sync_dist=True,batch_size=self.hparams.batch_size)
              
      return loss

  def validation_step(self, val_batch, batch_idx):
      imgs, targets = val_batch      
      loss_dict, preds = self(imgs, targets, mode='val')
      total_loss = 0
      for loss_name in loss_dict.keys():
      # ['loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg'])
        loss = loss_dict[loss_name]
        total_loss += loss
        self.log(f'val/{loss_name}', loss, sync_dist=True)
        # totloss = sum([loss for loss in loss_dict.values()])
  
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