import torch
import numpy as np
from urllib3.connectionpool import xrange
import torch.nn as nn

filepath = 'features2.txt'
file_list= []
with open(filepath) as fp:
    lines = [line.rstrip() for line in fp]

n_samples = len(lines)

c = 0
lbl_list = []
score_list = []

for i in range(n_samples):
    a = min(n_samples, i+53)
    for j in range(i+1, a):
        l1 = lines[i]
        l2 = lines[j]
        sepl1 = l1.split(',')
        sepl2 = l2.split(',')
        label1 = sepl1[1]
        label2 = sepl2[1]
        print(label1)
        # exit()
        # spll11 = label1.split('/')
        # label1 = spll11[-2]

        pair_lbl = 1
        if label1 == label2:
            pair_lbl = 0

        lbl_list.append(pair_lbl)

        features1 = sepl1[2:-1]
        features2 = sepl2[2:-1]

        features1 = [float(f) for f in features1]
        features2 = [float(f) for f in features2]


        f1_np = np.array(features1)
        f2_np = np.array(features2)


        distance = np.linalg.norm(f1_np-f2_np, ord=2, axis=0)


        # if c%1000==0:
        #     print(c, pair_lbl, distance)

        score_list.append(distance)
        c +=1


        if c>1000:
            from sklearn.metrics import roc_curve, auc
            import matplotlib.pyplot as plt

            y_score = np.array(score_list)
            y_lbl = np.array(lbl_list)

            # print(y_lbl.shape)
            # print(y_lbl.sum())

            # y0 = np.random.randn(5000, 1)
            # y1 = np.random.randn(5000, 1)+1
            # y_score = np.concatenate((y0, y1), 0)
            #
            # lbl0 = np.zeros([5000, 1])
            # lbl1 = np.ones([5000, 1])
            #
            #
            # y_lbl = np.concatenate((lbl0, lbl1), 0)

            fpr, tpr, _ = roc_curve(y_lbl, y_score)
            roc_auc = auc(fpr, tpr)

            print("area under the curve is:", roc_auc)

            plt.plot(fpr, tpr)
            plt.show()



        # if c>10:


        #     exit()
        # print(c)
        # if c>10000:




            #
            # # Compute micro-average ROC curve and ROC area
            # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
            # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            #
            #
            #
            #
            #
            # for a specific class
            #
            # plt.figure()
            # lw = 2
            # plt.plot(fpr[2], tpr[2], color='darkorange',
            #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
            # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('Receiver operating characteristic example')
            # plt.legend(loc="lower right")
            # plt.show()


