import numpy as np
import sklearn.svm as svm
from sklearn.linear_model import LogisticRegression
from conf_matrix import func_confusion_matrix
from data_preparation import prepare_data
from sklearn import metrics

# Prepare training and Test Data by splitting in training data
X_Train, X_Test, Y_Train, Y_Test = prepare_data(test_size=0.35, seed=0)

model1 = LogisticRegression(
    penalty='l1',
    C=4,
    fit_intercept=True,
    solver='liblinear',
    max_iter=100,
    l1_ratio=None
    )
model1.fit(X_Train, Y_Train)
predictions1 = model1.predict(X_Test)

model2 = svm.SVC(
    kernel='linear',
    C=4,
    gamma='auto',
    shrinking=True,
    max_iter=1919000000,
    decision_function_shape='ovo')
model2.fit(X_Train, Y_Train)
predictions2 = model2.predict(X_Test)

predictions_sum = predictions1 + predictions2
predictions = np.around(predictions_sum / 2)

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
