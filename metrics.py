#%%

# 计算各种评估指标

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import *

y_df = pd.read_csv('./all/prob_scores.csv',index_col=0)
y_real = y_df.values[:,0]
y_scores = y_df.values[:,1:]
r,c = y_df.shape
accuracy = 0
precision = 0
recall = 0
f1 = 0
auc = 0

for i in range(1, c):
    y_score = y_df.values[:,i]
    y_pred = np.zeros((len(y_real)))
    y_pred[y_score>0.55] = 1

    C = confusion_matrix(y_real, y_pred)
    precision += precision_score(y_real, y_pred)
    accuracy += accuracy_score(y_real, y_pred)
    recall += recall_score(y_real, y_pred)
    f1 += f1_score(y_real, y_pred)
    auc += roc_auc_score(y_real, y_score)
    

accuracy = accuracy / (c-1)
recall = recall / (c-1)
f1 = f1 / (c-1)
auc = auc / (c-1)
precision = precision / (c-1)
#%%

#drawPR(y_real, y_score)
# %%
def drawROC(y_pred, y_real, n_classes, PATH):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_real, y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_real.repeat(n_classes).ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    
    colors = ['blue','orange','green','red','purple','saddlebrown','deeppink','gray','gold','aqua','black']
    plt.figure(figsize=(8,8))


    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (AUC = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=2)
    
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (AUC = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=2)

    lw = 1
    for i in list(range(n_classes)):
        plt.plot(fpr[i], tpr[i], color=colors[i],
                lw=lw, label='division_{n}'.format(n=i) + '(AUC = {:.2f})'.format(roc_auc[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (protein interact prediction)')
    plt.legend(loc="lower right")
    plt.savefig(PATH ,dpi=600)

if c-1 > 5:
    n_lines = 5
else:
    n_lines = c-1
drawROC(y_scores[:,:n_lines], y_real, n_lines, './cr/roc_curve.png')
#%%
def drawLogs(log_path, PATH):
    lw = 2
    log_df = pd.read_csv(log_path, index_col=0)
    logs = log_df.values.T / 30
    colors = ['blue','orange','green','red']
    plt.figure(figsize=(12,8))
    plt.plot(logs[0],
            label='train_loss',
            color='deeppink', linestyle='--', linewidth=lw, alpha=0.5)
    plt.plot(logs[1],
            label='train_acc',
            color='orange', linewidth=lw, alpha=0.5)
    plt.plot(logs[2],
            label='test_loss',
            color='navy', linestyle='--', linewidth=lw, alpha=0.5)
    plt.plot(logs[3],
            label='test_acc',
            color='lightblue', linewidth=lw, alpha=0.5)
    plt.xlabel('epoch')
    #plt.ylim([0.5, 0.8])
    plt.title('Training Logs')
    plt.legend(loc="upper right")
    plt.savefig(PATH ,dpi=600)

drawLogs('./all/logs.csv', './all/logs.png')
# %%
