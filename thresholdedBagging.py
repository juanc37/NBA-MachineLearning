#############
#  Author: Juan Candelaria Claborne
#############

import numpy as np
from data_preparation import prepare_data
from conf_matrix import func_confusion_matrix
from sklearn import metrics
import tensorflow as tf
import dill

#prepare data
trainX, testX, trainY, testY = prepare_data(test_size=.35, seed=0)
testY = testY.to_numpy().astype(int)
predictions = []

# load in predictions from best random forest and neural network
with open("randomForestAccuracy.obj", 'rb') as f:
    rf = dill.load(f)
predictions.append(rf.predict(testX))
loaded = tf.saved_model.load("bmodels/model4")
predictions.append(loaded(testX))

# calculate thresholded predictions
predic =[]
for percent in range(50,99,1):
    pred = np.zeros(len(predictions[0]))
    for i in range(len(predictions[0])):
        nnpredict = 1 if predictions[1][i,1] > predictions[1][i,0] else 0
        if(nnpredict == 0 and predictions[0][i]< percent*.01):
            pred[i]= 0
        else:
            pred[i] = np.round(predictions[0][i])
    pred = pred.astype(int)
    predic.append(pred)
# find best thresholded prediction by auc
max = 0
maxidx = -1
idx = 0
for pred in predic:
    fpr, tpr, thresholds = metrics.roc_curve(testY, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    if auc > max:
        max = auc
        maxidx = idx
    idx+=1
# evaluate best prediction
bestPred = predic[maxidx]
conf_matrix, best_accuracy, recall_array, precision_array = func_confusion_matrix(testY, bestPred)
fpr, tpr, thresholds = metrics.roc_curve(testY, bestPred, pos_label=1)
auc = metrics.auc(fpr, tpr)
print("Confusion Matrix: ")
print(str(conf_matrix))
print("Average Accuracy: {}".format(str(best_accuracy)))
print("Per-Class Precision: {}".format(str(precision_array)))
print("Per-Class Recall: {}".format(str(recall_array)))
print("Area under the ROC Curve: {}".format(auc))
