import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "2" # put -1 to not use any

import torch
import numpy as np
import pandas as pd
import json 

from tqdm import tqdm
from torchvision.ops import nms
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.ops import masks_to_boxes

from utils import *
from evaluation import *
from visualization import *
from nets.segmentation import  Mask2Former
from loaders.hyst_dataset import BBXImageEvalTransform,BBXImageTrainTransform,BBXImageTestTransform, HystDataModuleFormer


concats = ['Bipolar', 'Vessel Sealer', 'Robot Grasper Heat', 'Scissors', 'Suction', 'Robot Scissors', 'monopolarhook' ]

def remove_empty_predictions(pred_masks):
  indices = []
  for idx in range(pred_masks.shape[0]):
    if pred_masks[idx].max() != 0:
      indices.append(idx)
  return indices

def construct_class_mapping(df_labels, class_column, label_column):
    df_labels = df_labels.loc[df_labels[label_column] != 'Needle']
    df_labels.loc[ df_labels[label_column].isin(concats), label_column ] = 'Others'

    unique_classes = sorted(df_labels[label_column].unique())
    class_mapping = {value: idx+1 for idx, value in enumerate(unique_classes)}
    return class_mapping

def remove_labels(df, class_mapping, class_column, label_column):

    df = df.loc[df['to_drop'] == 0]
    
    df = df.loc[df[label_column] != 'Needle']
    df.loc[ df[label_column].isin(concats), label_column ] = 'Others'

    df[class_column] = df[label_column].map(class_mapping)

    print(f"{df[[label_column, class_column]].drop_duplicates()}")
    return df.reset_index()

df_train = pd.read_csv('/CMF/data/lumargot/hysterectomy/mnt/surgery_tracking/csv/dataset_train_train.csv')
df_val = pd.read_csv('/CMF/data/lumargot/hysterectomy/mnt/surgery_tracking/csv/dataset_train_test.csv')
df_test = pd.read_csv('/CMF/data/lumargot/hysterectomy/mnt/surgery_tracking/csv/dataset_test.csv')

df_labels = pd.concat([df_train, df_val, df_test])

img_column = 'img_path'
seg_column = 'seg_path'
class_column = 'simplified_class'
label_column = 'simplified_label'
mnt = '/CMF/data/lumargot/hysterectomy/mnt/surgery_tracking/'

class_mapping = construct_class_mapping(df_labels, class_column, label_column)

df_test = remove_labels(df_test, class_mapping, class_column, label_column)
df_train = remove_labels(df_train, class_mapping, class_column, label_column)
df_val = remove_labels(df_val, class_mapping, class_column, label_column)


unique_classes = np.sort(np.unique(df_train['class']))
out_features = len(unique_classes)+1 # background

ttdata = HystDataModuleFormer( df_train, df_val, df_test, batch_size=1, num_workers=4, 
                            img_column=img_column,seg_column=seg_column, class_column=class_column, 
                            mount_point=mnt,train_transform=BBXImageTrainTransform(),
                            valid_transform=BBXImageEvalTransform(), test_transform=BBXImageTestTransform())

ttdata.setup()
test_dl = ttdata.test_dataloader()
ds = ttdata.test_ds

# Create model
ckpt = '/CMF/data/lumargot/hysterectomy/out/mask2former/epoch=17-val_loss=40.57.ckpt'
model = Mask2Former.load_from_checkpoint(ckpt)
model.cuda()
model.eval()

y_true, y_pred = [], []
stats = defaultdict(list)
data_dir = os.path.splitext(ckpt)[0]
SCORE_THR = 0.5

with torch.no_grad():
  for idx, batch in tqdm(enumerate(test_dl), total=len(test_dl)): 
    for k,v in batch.items():
      if isinstance(batch[k], list) :
        batch[k] = [ elt.cuda() for elt in batch[k]] 
      else:
        batch[k] = batch[k].cuda()

    gt_masks = batch['mask_labels'][0].cpu().detach()
    gt_labels = batch['class_labels'][0].cpu().detach()
    gt_boxes = boxes = torch.tensor(ds[idx]['boxes'])
    _, outs = model(batch)

    original_sizes = [(img.shape[1],img.shape[2]) for img in batch['pixel_values']]  

    # Post-process outputs
    result= model.processor.post_process_instance_segmentation( outs, target_sizes=original_sizes,)[0]

    multimask = result['segmentation']
    segments = result['segments_info']

    pred_masks, pred_labels,  pred_scores = [], [], []

    if len(segments) > 0:
      for segment in segments:
        id, label, score = segment['id'], segment['label_id'], segment['score']
        binary_mask = (result['segmentation'] == id).unsqueeze(0).to(torch.float32).cpu().detach()
        
        pred_masks.append(binary_mask)
        pred_labels.append(torch.tensor(label).unsqueeze(0))
        pred_scores.append(torch.tensor(score).unsqueeze(0))
        
      pred_masks = torch.cat(pred_masks)
      pred_labels = torch.cat(pred_labels)
      pred_scores = torch.cat(pred_scores)

      gt_dic = {'boxes':gt_boxes, 'labels':gt_labels, 'masks':gt_masks}
      pred_dic = {'scores':pred_scores, 'labels':pred_labels, 'masks': pred_masks}

      if (pred_scores >=SCORE_THR).any():
        try:
          
          keep = pred_scores >= SCORE_THR
          pred_dic=apply_indices_selection(pred_dic, keep)

          keep = remove_empty_predictions(pred_masks)
          pred_dic=apply_indices_selection(pred_dic, keep)
          pred_boxes = masks_to_boxes(pred_dic['masks'])

          pred_dic['boxes'] = pred_boxes

          nms_indices = nms(pred_dic['boxes'], pred_dic['scores'],0.3)
          pred_dic_filtered=apply_indices_selection(pred_dic, nms_indices)

          stats, gt_label, pred_label = get_prediction_metrics(gt_dic, pred_dic_filtered, stats, iou_threshold=0.25)
          y_true.append(torch.tensor(gt_label))
          y_pred.append(torch.tensor(pred_label))
        except:
           print("error")

class_names = df_test['simplified_label'].unique()
class_names = list(class_mapping.keys())
y_true = torch.concat(y_true)
y_pred = torch.concat(y_pred)

df_pred, out_dict = compute_global_metrics(class_names, y_true, y_pred, stats, iou_threshold=0.25)

if not os.path.exists(data_dir):
    os.mkdir(data_dir)
df_pred.to_csv(os.path.join(data_dir, 'prediction.csv'))
filename = os.path.join(data_dir, 'output_stats.json')
with open(filename, 'w') as file:
    json.dump(out_dict, file, indent=2)

report = classification_report(y_true, y_pred, output_dict=True)
print(report)

print(json.dumps(report, indent=2))
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(os.path.join(data_dir, 'classification_report.csv'))

cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=3)

# Plot non-normalized confusion matrix
fig = plt.figure(figsize=(16,6))

plt.subplot(121)
plot_confusion_matrix(cnf_matrix, classes=class_names, title='confusion matrix')

plt.subplot(122)
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='confusion matrix - normalized')

plt.savefig(os.path.join(data_dir, 'confusion_matrix.png'))