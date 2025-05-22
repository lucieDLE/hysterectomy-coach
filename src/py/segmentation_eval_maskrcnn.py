import os 
os.environ['CUDA_VISIBLE_DEVICES']="0"
import argparse
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision.ops import nms
from sklearn.metrics import classification_report, confusion_matrix

from utils import *
from evaluation import *
from visualization import *

from nets.segmentation import MaskRCNN, FasterRCNN
from loaders.hyst_dataset import HystDataModuleSeg, BBXImageTestTransform, BBXImageTrainTransform, BBXImageEvalTransform


concats = [ 'Bipolar', 'Vessel Sealer', 'Scissors', 'Suction', 'Robot Scissors', 'monopolarhook' ]


def construct_class_mapping(df_labels, class_column, label_column):
    df_labels = df_labels.loc[df_labels[label_column] != 'Needle']
    df_labels.loc[ df_labels[label_column].isin(concats), label_column ] = 'Others'

    unique_classes = sorted(df_labels[label_column].unique())
    class_mapping = {value: idx+1 for idx, value in enumerate(unique_classes)}
    return class_mapping

def remove_labels(df, class_mapping, class_column, label_column):

    df = df.loc[df['to_drop'] == 0] # remove small boxes
    
    df = df.loc[df[label_column] != 'Needle']
    df.loc[ df[label_column].isin(concats), label_column ] = 'Others'

    df[class_column] = df[label_column].map(class_mapping)

    print(f"{df[[label_column, class_column]].drop_duplicates()}")
    return df.reset_index()


def main(args):

    df_test = pd.read_csv('/CMF/data/lumargot/hysterectomy/mnt/surgery_tracking/csv/dataset_test.csv')
    df_train = pd.read_csv('/CMF/data/lumargot/hysterectomy/mnt/surgery_tracking/csv/dataset_train_train.csv')
    df_val = pd.read_csv('/CMF/data/lumargot/hysterectomy/mnt/surgery_tracking/csv/dataset_train_test.csv')

    df_labels = pd.concat([df_train, df_val, df_test])

    SCORE_THR = args.score_thr

    class_mapping = construct_class_mapping(df_labels, args.class_column, args.label_column)

    df_test = remove_labels(df_test, class_mapping, args.class_column, args.label_column)
    df_train = remove_labels(df_train, class_mapping, args.class_column, args.label_column)
    df_val = remove_labels(df_val, class_mapping, args.class_column, args.label_column)


    ttdata = HystDataModuleSeg( df_test, df_test, df_test, batch_size=1, num_workers=1, 
                                img_column=args.img_column,seg_column=args.seg_column, class_column=args.class_column, 
                                mount_point=args.mount_point,train_transform=BBXImageTrainTransform(),valid_transform=BBXImageEvalTransform(), 
                                test_transform=BBXImageTestTransform())

    ttdata.setup()

    test_dl = ttdata.test_dataloader()
    class_names = list(class_mapping.keys())

    data_dir = os.path.splitext(args.model)[0]

    model = MaskRCNN.load_from_checkpoint(args.model)
        
    model.eval()
    model.cuda()

    y_true, y_pred = [], []
    stats = defaultdict(list)

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_dl), total=len(test_dl)):     
            imgs = []
            img, batch = batch

            imgs.append(img)
            imgs = torch.cat(imgs)

            outputs = model.forward(imgs.cuda(), mode='test')
            img_np = 255*imgs[0].permute(1,2,0).numpy()
            
            pred_boxes = outputs[0]['boxes'].cpu().detach()
            pred_masks = outputs[0]['masks'].cpu().detach()

            # convert mask to binary
            pred_masks = convert_to_binary_mask(pred_masks)
            pred_labels = outputs[0]['labels'].cpu().detach()
            pred_scores = outputs[0]['scores'].cpu().detach()

            gt_masks = batch[0]['masks']
            gt_boxes = ttdata.compute_bb_mask(gt_masks, pad=0.01).numpy()
            gt_labels = batch[0]['labels'].cpu().detach().numpy()

            pred_dic = {'boxes':pred_boxes, 'scores':pred_scores, 'labels':pred_labels, 'masks': pred_masks}
            gt_dic = {'boxes':gt_boxes, 'labels':gt_labels, 'masks':gt_masks}

            if (pred_scores >=SCORE_THR).any():
                
                keep = pred_scores >= SCORE_THR
                pred_dic=apply_indices_selection(pred_dic, keep)

                nms_indices = nms(pred_dic['boxes'], pred_dic['scores'],0.3)
                pred_dic_filtered=apply_indices_selection(pred_dic, nms_indices)

                indices_keep = select_indices_to_keep(pred_dic_filtered['labels'], pred_dic_filtered['boxes'], pred_dic_filtered['scores'])
                pred_dic_filtered=apply_indices_selection(pred_dic_filtered, indices_keep)

                # get prediction metrids individually -> can't read all masks in memory
                stats, gt_label, pred_label = get_prediction_metrics(gt_dic, pred_dic_filtered, stats, iou_threshold=0.25)
                y_true.append(torch.tensor(gt_label))
                y_pred.append(torch.tensor(pred_label))

    y_true = torch.concat(y_true)
    y_pred = torch.concat(y_pred)

    # compute global metrics
    df_pred, out_dict = compute_global_metrics(class_names, y_true, y_pred, stats, iou_threshold=0.25)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    df_pred.to_csv(os.path.join(data_dir, 'prediction.csv'))
    
    filename = os.path.join(data_dir, 'output_stats.json')
    with open(filename, 'w') as file:
        json.dump(out_dict, file, indent=2)
    print(out_dict)

    report = classification_report(y_true, y_pred, output_dict=False)
    print(report)

    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(data_dir, 'classification_report.csv'))


    cnf_matrix = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(16,6))
    plt.subplot(121)
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='confusion matrix')
    plt.subplot(122)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='confusion matrix - normalized')
    plt.savefig(os.path.join(data_dir, 'confusion_matrix.png'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Surgical Tool Segmentation Training')
    
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--model', help='Model path to continue training', type=str, default=None)
    input_group.add_argument('--img_column', help='Image/video column name', type=str, default='img_path')
    input_group.add_argument('--seg_column', help='segmentation column name', type=str, default='seg_path')
    input_group.add_argument('--class_column', help='Class column', type=str, default='class')
    input_group.add_argument('--label_column', help='label column', type=str, default='Instrument Name')
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    input_group.add_argument('--score_thr', help='confidence score threshold to select best predictions', type=float, default=0.4)

    args = parser.parse_args()
    main(args)