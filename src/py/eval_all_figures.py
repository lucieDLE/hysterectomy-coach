import numpy as np
import argparse
import importlib
import os
from datetime import datetime
import json
import pdb
import glob
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score
import pdb
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
    parser.add_argument('--figsize', type=float, nargs='+', help='Figure size', default=(16, 12))
    parser.add_argument('--title', type=str, help='Title for the image', default="Confusion matrix")

    return parser



def update_axes_ranges(fig,dx):
    fig.update_xaxes(range=[0-dx, 1+dx])
    fig.update_yaxes(range=[0-dx, 1+dx])
    return fig

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
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

    dx = 0.05
    
    df = pd.read_csv(args.csv)

    df_names = df.sort_values(by=["tag"])

    class_names = df_names[[args.csv_tag_column, args.csv_prediction_column]][args.csv_tag_column].drop_duplicates()

    for idx, row in df.iterrows():
        y_true_arr.append(row[args.csv_true_column])
        y_pred_arr.append(row[args.csv_prediction_column])


    #### ------------ 1. Confusion Matrix --------------- ###
    cnf_matrix = confusion_matrix(y_true_arr, y_pred_arr)
    np.set_printoptions(precision=3)

    # Plot non-normalized confusion matrix
    fig1 = plt.figure(figsize=args.figsize)
    plot_confusion_matrix(cnf_matrix, classes=class_names, title=args.title)
    confusion_filename = os.path.splitext(args.csv)[0] + "_confusion.png"
    fig1.savefig(confusion_filename)

    # Plot normalized confusion matrix
    fig2 = plt.figure(figsize=args.figsize)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title=args.title + ' - normalized')
    
    norm_confusion_filename = os.path.splitext(args.csv)[0] + "_norm_confusion.png"
    fig2.savefig(norm_confusion_filename)

    ### ---- ROC, PR and microaveraged Curves 

    probs_fn = args.csv.replace("_prediction.csv", "_probs.pickle")
    print(probs_fn, os.path.splitext(probs_fn)[1])
    
    if os.path.exists(probs_fn) and os.path.splitext(probs_fn)[1] == ".pickle":
      with open(probs_fn, 'rb') as f:
        y_scores = pickle.load(f)
    
    y_onehot = pd.get_dummies(y_true_arr)
    
    ### ------------- 3. ROC curve ---------------
    

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fig3 = go.Figure()
    fig3.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=1, y0=0, y1=1)

    for i in range(y_onehot.shape[1]):

        y_true = y_onehot.iloc[:, i]
        y_score = y_scores[:, i]

        fpr[i], tpr[i], _ = roc_curve(y_true, y_score)
        roc_auc[i] = roc_auc_score(y_true, y_score)

        name = f"{y_onehot.columns[i]} (AUC={roc_auc[i]:.2f})"
        fig3.add_trace(go.Scatter(x=fpr[i], y=tpr[i], name=name, mode='lines'))

    fig3.update_layout(title={'text': 'Per-class ROC curves','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'},
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=900, height=700
    )
    fig3 = update_axes_ranges(fig3,dx)

    roc_filename = os.path.splitext(args.csv)[0] + "_roc.png"
    fig3.write_image(roc_filename)


    ### ----------- 4. micro-averaged ROC Curve ----------

    auc_score = roc_auc_score(y_onehot, y_scores, average='micro')

    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot.to_numpy().ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    fig4 = go.Figure()
    fig4.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=1, y0=0, y1=1)

    name = f"micro-average ROC curve={auc_score:.2f}"
    fig4.add_trace(go.Scatter(x=fpr["micro"], y=tpr["micro"], name=name, mode='lines'))
    fig4.update_layout(title={'text': name,'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'},
                      xaxis_title='False Positive Rate',yaxis_title='True Positive Rate',
                      yaxis=dict(scaleanchor="x", scaleratio=1),xaxis=dict(constrain='domain'),
                      width=900, height=700
                      )
    fig4 = update_axes_ranges(fig4,dx)

    uroc_filename = os.path.splitext(args.csv)[0] + "_micro_roc.png"

    fig4.write_image(uroc_filename)


    ### ----------- 5. PR Curves ----------
    import pdb
    precision = dict()
    recall = dict()
    average_precision = dict()

    fig5 = go.Figure()
    fig5.add_shape(type='line', line=dict(dash='dash'),x0=1, x1=0, y0=0, y1=1)

    for i in range(y_onehot.shape[1]):

        y_true = y_onehot.iloc[:, i]
        y_score = y_scores[:, i]

        precision[i], recall[i], _ = precision_recall_curve(y_true, y_score)
        average_precision[i] = average_precision_score(y_true, y_score)

        name = f"{y_onehot.columns[i]} (AP={average_precision[i]:.2f})"
        fig5.add_trace(go.Scatter(x=recall[i], y=precision[i], name=name, mode='lines'))

    fig5.update_layout(title={'text': 'Per-class PR curves','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'},
                       xaxis_title='Recall',yaxis_title='Precision',
                       yaxis=dict(scaleanchor="x", scaleratio=1),xaxis=dict(constrain='domain'),
                       width=900, height=700)
    fig5 = update_axes_ranges(fig5,dx)

    pr_filename = os.path.splitext(args.csv)[0] + "_pr.png"
    fig5.write_image(pr_filename)


    ### ----------- 6. micro-averaged PR Curve ----------

    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(y_true, y_score, average="micro")

    fig6 = go.Figure()
    fig6.add_shape(type='line', line=dict(dash='dash'),x0=1, x1=0, y0=0, y1=1)

    name = f"micro-average PR curve={average_precision['micro']:.2f}"

    fig6.add_trace(go.Scatter(x=recall["micro"], y=precision["micro"], name=name, mode='lines'))
    fig6.update_layout(title={'text': name,'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'},
                      xaxis_title='Recall',yaxis_title='Precision',
                      yaxis=dict(scaleanchor="x", scaleratio=1),xaxis=dict(constrain='domain'),
                      width=900, height=700
                      )
    
    fig6 = update_axes_ranges(fig6,dx)

    upr_filename = os.path.splitext(args.csv)[0] + "_micro_pr.png"

    fig6.write_image(upr_filename)


if __name__ == "__main__":
  
  parser = get_argparse()
  
  args = parser.parse_args()

  main(args)