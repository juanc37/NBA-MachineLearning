#############
#  Author: Caleb Gelnar
#############

import sklearn.svm as svm
from conf_matrix import func_confusion_matrix
from data_preparation import prepare_data
from sklearn import metrics

## Load data
X_Train, X_Test, Y_Train, Y_Test = prepare_data(test_size=0.35, seed=0)

model = svm.SVC(
    kernel='linear',
    C=4,
    gamma='auto',
    shrinking=True,
    max_iter=1919000000,
    decision_function_shape='ovo')
model.fit(X_Train, Y_Train)
predictions = model.predict(X_Test)

## Evaluate your results in terms of accuracy, real, or precision.
conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(Y_Test, predictions)
fpr, tpr, thresholds = metrics.roc_curve(Y_Test, predictions, pos_label=1)
auc = metrics.auc(fpr, tpr)
print()
print("########### MODEL PERFORMANCE ###########")
print("Confusion Matrix: ")
print(conf_matrix)
print("Average Accuracy: {}".format(accuracy))
print("Per-Class Precision: {}".format(precision_array))
print("Per-Class Recall: {}".format(recall_array))
print("#########################################")
print("Area under the ROC Curve: {}".format(auc))
