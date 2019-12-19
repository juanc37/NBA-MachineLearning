#############
#  Author: Caleb Gelnar
#############

import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.model_selection import train_test_split
from conf_matrix import func_confusion_matrix
from data_preparation import prepare_data
from sklearn import metrics

## Load data
X_Train, X_Test, Y_Train, Y_Test = prepare_data(test_size=0.35, seed=0)

# Subsets for training models including validation subset (test_size is for validation set)
X_Train, X_Validation, Y_Train, Y_Validation = train_test_split(X_Train, Y_Train, test_size=0.25, random_state=0)

## Model selection over validation set
# consider the parameters C and kernel types (linear, RBF, and poly)
# Plot the validation errors while using different values of C (with other hyper-parameters fixed)
c_range = [.001, .01, .1, 1, 2, 4, 8, 16]  #
svm_c_error = []

for c_value in c_range:
    model = svm.SVC(kernel='linear', C=c_value, gamma='auto')
    model.fit(X_Train, Y_Train)
    error = 1. - model.score(X_Validation, Y_Validation)
    svm_c_error.append(error)

plt.plot(c_range, svm_c_error)
plt.title('Linear SVM')
plt.xlabel('c values')
plt.ylabel('error')
plt.show()


# Plot the validation errors while using linear, and Polynomial kernel (with other hyper-parameters fixed)
kernel_types = ['linear', 'poly', 'rbf']
svm_kernel_error = []
for kernel_value in kernel_types:
    model = svm.SVC(kernel=kernel_value, C=1, gamma='auto')
    model.fit(X_Train, Y_Train)
    error = 1. - model.score(X_Validation, Y_Validation)
    svm_kernel_error.append(error)

plt.plot(kernel_types, svm_kernel_error)
plt.title('SVM by Kernels')
plt.xlabel('Kernel')
plt.ylabel('error')
plt.xticks(kernel_types)
plt.show()

## Select the best model and apply it over the testing subset
best_kernel = kernel_types[svm_kernel_error.index(min(svm_kernel_error))]
best_c = c_range[svm_c_error.index(min(svm_c_error))]
model = svm.SVC(kernel=best_kernel, C=best_c, gamma='auto')
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
