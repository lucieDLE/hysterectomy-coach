

import numpy as np
import argparse
import importlib
import os
from datetime import datetime
import json
import glob
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools
from scipy import interp
import pickle 

import plotly.graph_objects as go
import plotly.express as px


def get_argparse():
    parser = argparse.ArgumentParser(description='Evaluate classification result', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--csv', type=str, help='csv file', required=True)
    parser.add_argument('--csv_true_column', type=str, help='Which column to do the stats on', default="tag")
    parser.add_argument('--csv_tag_column', type=str, help='Which column has the actual names', default="class")
    parser.add_argument('--csv_prediction_column', type=str, help='csv true class', default='pred')
    parser.add_argument('--title', type=str, help='Title for the image', default="Confusion matrix")
    parser.add_argument('--figsize', type=float, nargs='+', help='Figure size', default=(16, 12))

    return parser


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.xlabel('Predicted label')
  plt.ylabel('True label')
  plt.tight_layout()

  return cm


def main(args):
    y_true_arr = [] 
    y_pred_arr = []


    if(os.path.splitext(args.csv)[1] == ".csv"):        
        df = pd.read_csv(args.csv)
    else:        
        df = pd.read_parquet(args.csv)

    class_names = df[[args.csv_tag_column, args.csv_prediction_column]][args.csv_tag_column].drop_duplicates()

    for idx, row in df.iterrows():
        y_true_arr.append(row[args.csv_true_column])
        y_pred_arr.append(row[args.csv_prediction_column])


    # Compute confusion matrix
    class_names = pd.Series({0: 'g-j a', 1:'dsi', 2:'leak t', 3:'his', 4:'ot', 5:'lt', 6:'j-j', 7:'ft', 8:'bp limb', 9:'loa', 10:'pc', 11:'md', 12:'pd', 13:'rny'})
    cnf_matrix = confusion_matrix(y_true_arr, y_pred_arr)
    np.set_printoptions(precision=3)

    # Plot non-normalized confusion matrix
    fig = plt.figure(figsize=args.figsize)
    plot_confusion_matrix(cnf_matrix, classes=class_names, title=args.title)
    confusion_filename = os.path.splitext(args.csv)[0] + "_confusion.png"
    fig.savefig(confusion_filename)

    # Plot normalized confusion matrix
    fig2 = plt.figure(figsize=args.figsize)
    cm = plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title=args.title + ' - normalized')



    # cnf_matrix = np.array(cnf_matrix)
    # FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    # FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    # TP = np.diag(cnf_matrix)
    # TN = cnf_matrix.values.sum() - (FP + FN + TP)

    # # Sensitivity,hit rate, recall, or true positive rate
    # TPR = TP/(TP+FN)
    # # Specificity or true negative rate
    # TNR = TN/(TN+FP) 
    # # Precision or positive predictive value
    # PPV = TP/(TP+FP)
    # # Negative predictive value
    # NPV = TN/(TN+FN)
    # # Fall out or false positive rate
    # FPR = FP/(FP+TN)
    # # False negative rate
    # FNR = FN/(TP+FN)
    # # False discovery rate
    # FDR = FP/(TP+FP)

    # # Overall accuracy
    # ACC = (TP+TN)/(TP+FP+FN+TN)

    # print("True positive rate:", TPR)
    # print("True negative rate:", TNR)
    # print("Precision or positive predictive value:", PPV)
    # print("Negative predictive value:", NPV)
    # print("False positive rate or fall out", FPR)
    # print("False negative rate:", FNR)
    # print("False discovery rate:", FDR)
    # print("Overall accuracy:", ACC)

    print(classification_report(y_true_arr, y_pred_arr))
    report = classification_report(y_true_arr, y_pred_arr,output_dict=True)

    norm_confusion_filename = os.path.splitext(args.csv)[0] + "_norm_confusion.png"
    fig2.savefig(norm_confusion_filename)


    probs_fn = args.csv.replace("_prediction.csv", "_probs.pickle")
    print(probs_fn, os.path.splitext(probs_fn)[1])
    if os.path.exists(probs_fn) and os.path.splitext(probs_fn)[1] == ".pickle":
      
      print("Reading:", probs_fn)

      with open(probs_fn, 'rb') as f:
        y_scores = pickle.load(f)
    
    y_onehot = pd.get_dummies(y_true_arr)
    
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    Y1 = y_scores[:, 0:9]
    Y2 = y_scores[:,10:]
    y_scores = np.concatenate((Y1,Y2), axis=1)

    import pdb
    for i in range(y_onehot.shape[1]):
        print(i)

        y_true = y_onehot.iloc[:, i]
        y_score = y_scores[:, i]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)
        # roc_auc = auc(fpr, tpr)

        report[str(i)]["auc"] = auc_score

        name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=900, height=700
    )
    # Compute micro-average ROC curve and ROC area
    # pdb.set_trace()
    # auc_score = roc_auc_score(y_onehot, y_scores, average='micro')

    # fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot.to_numpy().ravel(), y_scores.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # fig = go.Figure()
    # fig.add_shape(
    #     type='line', line=dict(dash='dash'),
    #     x0=0, x1=1, y0=0, y1=1
    # )
    # name = f"micro-average ROC curve={auc_score:.2f}"
    # fig.add_trace(go.Scatter(x=fpr["micro"], y=tpr["micro"], name=name, mode='lines'))
    # fig.update_layout(
    #     title={
    #     'text': name,
    #     'y':0.9,
    #     'x':0.5,
    #     'xanchor': 'center',
    #     'yanchor': 'top'},
    #     xaxis_title='False Positive Rate',
    #     yaxis_title='True Positive Rate',
    #     yaxis=dict(scaleanchor="x", scaleratio=1),
    #     xaxis=dict(constrain='domain'),
    #     width=900, height=700
    # )

    roc_filename = os.path.splitext(args.csv)[0] + "_roc.png"

    fig.write_image(roc_filename)

    # support = []
    # auc = []
    # for i in range(y_scores.shape[1]):
    #     support.append(report[str(i)]["support"])
    #     auc.append(report[str(i)]["auc"])

    # support = np.array(support)
    # auc = np.array(auc)

    # report["macro avg"]["auc"] = np.average(auc) 
    # report["weighted avg"]["auc"] = np.average(auc, weights=support) 
        
    # df_report = pd.DataFrame(report).transpose()
    # report_filename = os.path.splitext(args.csv)[0] + "_classification_report.csv"
    # df_report.to_csv(report_filename)



if __name__ == "__main__":
  
  parser = get_argparse()
  
  args = parser.parse_args()

  main(args)