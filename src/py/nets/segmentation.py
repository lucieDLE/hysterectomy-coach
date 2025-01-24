from torchvision import models
import torch
import pytorch_lightning as pl
from tqdm import tqdm

class MaskRCNN(pl.LightningModule):
  def __init__(self, num_classes=4, **kwargs):
    super(MaskRCNN, self).__init__()        
    
    self.save_hyperparameters()
    
    self.model = models.detection.maskrcnn_resnet50_fpn(weights=models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    self.num_classes = num_classes
    in_features = self.model.roi_heads.box_predictor.cls_score.in_features
    self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    self.model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

  def configure_optimizers(self):
    # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.0001)
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