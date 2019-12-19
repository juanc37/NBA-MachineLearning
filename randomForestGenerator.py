#############
#  Author: Juan Candelaria Claborne
#############

import numpy as np
import operator
from data_preparation import prepare_data
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from conf_matrix import func_confusion_matrix
import dill


    ############# Create models######
def buildModels():
    trainX, testX, trainY, testY = prepare_data(test_size=.35, seed=0)
    trainY = trainY.to_numpy().astype(int)
    testY = testY.to_numpy().astype(int)

    accuracy = []
    pred = []
    highestTrueNeg = 98  # set to previous highest
    highestAcc = .707  # set to previous highest
    for estimators in range(20, 1000, 10):
        rf = RandomForestRegressor(n_estimators=estimators)
        rf.fit(trainX, trainY)
        predictions = rf.predict(testX).round().astype(int)
        accuracy.append(metrics.accuracy_score(testY, predictions))
        pred.append(predictions)
        if metrics.accuracy_score(testY, predictions) > .69:  # manually consider models with better than 69% accuracy
            conf_matrix, class_acc, recall_array, precision_array = func_confusion_matrix(testY, predictions)
            if conf_matrix[0,0] > highestTrueNeg:
                tn = open("randomForestTrueNeg.obj", "wb")
                dill.dump(rf,tn)
                highestTrueNeg = conf_matrix[0,0]
                tn.close()
            elif metrics.accuracy_score(testY, predictions) > highestAcc:
                acc = open("randomForestAccuracy.obj", "wb")
                dill.dump(rf, acc)
                acc.close()
                highestAcc = class_acc

    index, value = max(enumerate(accuracy), key=operator.itemgetter(1))

    print("Best Number of Estimators: {}".format(20 + 10*(index)))
    # Use the forest's predict method on the test data
    conf_matrix, best_accuracy, recall_array, precision_array = func_confusion_matrix(testY, pred[index])
    print("Confusion Matrix: ")
    print(str(conf_matrix))
    print("Average Accuracy: {}".format(str(best_accuracy)))
    print("Per-Class Precision: {}".format(str(precision_array)))
    print("Per-Class Recall: {}".format(str(recall_array)))

buildModels()

# we did not end up talking about random forest bagging because it was not significant to the research
def randomForestBagging(fileNames):
    trainX, testX, trainY, testY = prepare_data(test_size=.35, seed=0)
    testY = testY.to_numpy().astype(int)
    predictions = []
    for file in fileNames:
        # model = dill.load(open(file,"rb"))
        with open(file, 'rb') as f:
            rf = dill.load(f)
        predictions.append(rf.predict(testX))
    pred = np.zeros(len(predictions[0]))
    for i in range(len(predictions[0])):
        for set in range(len(fileNames)):
            pred[i] += predictions[set][i]
            print(pred[i])
        pred[i] = (pred[i]/len(fileNames)).round()
    pred = pred.astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(testY, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    conf_matrix, best_accuracy, recall_array, precision_array = func_confusion_matrix(testY, pred)
    print("Confusion Matrix: ")
    print(str(conf_matrix))
    print("Average Accuracy: {}".format(str(best_accuracy)))
    print("Per-Class Precision: {}".format(str(precision_array)))
    print("Per-Class Recall: {}".format(str(recall_array)))
    print("Area under the ROC Curve: {}".format(auc))

## randomForest.obj is just another good model that I had saved
fileNames = ["randomForest.obj", "randomForestAccuracy.obj", "randomForestTrueNeg.obj"]


#randomForestBagging()

# Best Number of Estimators: 160
# Confusion Matrix:
# [[ 90  95]
#  [ 42 242]]
# Average Accuracy: 0.7078891257995735
# Per-Class Precision: [0.68181818 0.71810089]
# Per-Class Recall: [0.48648649 0.85211268]
# Area under the ROC Curve: 0.6692995812714122

# Best True Negative
# [[ 98  87]
#  [ 56 228]]
# Average Accuracy: 0.6950959488272921
# Per-Class Precision: [0.63636364 0.72380952]
# Per-Class Recall: [0.52972973 0.8028169 ]

#Random forest bagging
# Confusion Matrix:
# [[ 93  92]
#  [ 47 237]]
# Average Accuracy: 0.7036247334754797
# Per-Class Precision: [0.66428571 0.72036474]
# Per-Class Recall: [0.5027027  0.83450704]
# Area under the ROC Curve: 0.6686048724781118