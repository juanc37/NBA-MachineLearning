#############
#  Author: Caleb Gelnar
#############

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from conf_matrix import func_confusion_matrix
from data_preparation import prepare_data
from sklearn import metrics

# Prepare training and Test Data by splitting in training data
X_Train, X_Test, Y_Train, Y_Test = prepare_data(test_size=0.5, seed=0)

# Subsets for training models including validation subset (test_size is for validation set)
X_Train, X_Validation, Y_Train, Y_Validation = train_test_split(X_Train, Y_Train, test_size=0.25, random_state=0)

# Train Model for many C values and visualize validation error (with other hyper-parameters fixed)
c_range = [.01, .1, 1, 2, 4, 8, 16, 32]
logistic_error = []
for c_value in c_range:
    model = LogisticRegression(
        penalty='l1',
        C=c_value,
        fit_intercept=True,
        solver='liblinear',
        max_iter=100,
        l1_ratio=None
    )
    model.fit(X_Train, Y_Train)
    error = 1. - model.score(X_Validation, Y_Validation)
    logistic_error.append(error)

plt.plot(c_range, logistic_error)
plt.title('Logistic Regression')
plt.xlabel('c values')
plt.ylabel('error')
plt.show()

# Train model for many penalty types and visualize validation error (with other hyper-parameters fixed)
penalty_types = ['l1', 'l2', 'elasticnet']
logistic_penalty_error = []
for penalty in penalty_types:
    if penalty == 'elasticnet':
        solver = 'saga'
        l1_ratio = 0
        max_iter = 10000
    else:
        solver = 'liblinear'
        l1_ratio = None
        max_iter = 100
    model = LogisticRegression(
        penalty=penalty,
        C=2,
        fit_intercept=True,
        solver=solver,
        max_iter=max_iter,
        l1_ratio=l1_ratio
    )
    model.fit(X_Train, Y_Train)
    error = 1. - model.score(X_Validation, Y_Validation)
    logistic_penalty_error.append(error)

plt.plot(penalty_types, logistic_penalty_error)
plt.title('Logistic Regression by Penalty')
plt.xlabel('Penalty')
plt.ylabel('error')
plt.xticks(penalty_types)
plt.show()

# Evaluate results in terms of accuracy, real, or precision with the best model.
best_penalty = penalty_types[logistic_penalty_error.index(min(logistic_penalty_error))]
best_c = c_range[logistic_error.index(min(logistic_error))]
print(best_penalty)
print(best_c)
if best_penalty == 'elasticnet':
    solver = 'saga'
    l1_ratio = 0
    max_iter = 10000
else:
    solver = 'liblinear'
    l1_ratio = None
    max_iter = 100
model = LogisticRegression(
    penalty=best_penalty,
    C=best_c,
    fit_intercept=True,
    solver=solver,
    max_iter=max_iter,
    l1_ratio=l1_ratio
    )
model.fit(X_Train, Y_Train)
predictions = model.predict(X_Test)
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
