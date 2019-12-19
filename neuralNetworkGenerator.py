#############
#  Author: Juan Candelaria Claborne
#############

import tensorflow as tf
import itertools
import sys
from conf_matrix import func_confusion_matrix
from data_preparation import prepare_data
import tensorflow.python.util.deprecation as deprecation
from sklearn import metrics

deprecation._PRINT_DEPRECATION_WARNINGS = False

#############
# Create data
trainX, testX, trainY, testY = prepare_data(test_size=.35, seed=0)
trainY = trainY.to_numpy().astype(int)
testY = testY.to_numpy().astype(int)


train = tf.data.Dataset.from_tensor_slices((trainX, trainY)) \
    .shuffle(len(trainY)) \
    .batch(16)


#############
# Make neural network
tf.keras.backend.set_floatx('float64')
modelCount = 0
activation = ['relu' , 'sigmoid', 'tanh', 'elu']
hiddenLayers = [3, 4, 5, 6, 7]
unitCounts = [4, 8, 12, 20, 28, 32]
epochs = [1,2,3,4]
optimizers = [tf.keras.optimizers.SGD(learning_rate=.5), 'adam']

f = open("results.txt", "w")
##compile combination of models
for layerCount in hiddenLayers:
    ## create all possible combinations of layers with n units and activations specified above
    activationCombos = list(itertools.product(activation, repeat=layerCount))
    unitCombos = list(itertools.product(unitCounts, repeat=layerCount))
    for activ in activationCombos:
        for unit in unitCombos:
            for e in epochs:
                model = tf.keras.Sequential()
                model.add(tf.keras.layers.Dense(units=20, activation='relu'))
                #  add the specified model's hidden layers
                for i in range(layerCount):
                    model.add(tf.keras.layers.Dense(units=unit[i], activation=activ[i]))
                model.add(tf.keras.layers.Dense(units=2, activation='softmax'))

                optimizers = [tf.keras.optimizers.SGD(learning_rate=.5), 'adam']
                for opt in optimizers:
                    model.compile(optimizer=opt,
                                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                                  verbose=0)

                    hist1 = model.fit(
                        train.repeat(),
                        epochs=e,
                        steps_per_epoch=250,
                        verbose=0
                    )
                    # evaluate model and save if it is good
                    predictions = model.predict(testX)
                    pred = []
                    for a in predictions:
                        pred.append(int(1 if a[1] >= a[0] else 0))
                    conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(testY, pred)
                    if (accuracy > .675 or accuracy < .325):
                        fpr, tpr, thresholds = metrics.roc_curve(testY, pred, pos_label=1)
                        auc = metrics.auc(fpr, tpr)
                        print(accuracy)
                        tf.saved_model.save(model, "bmodels/model" + str(modelCount))
                        f.write("model {} has accuracy of: {}".format(str(modelCount), str(accuracy)))
                        f.write("##################\n{}\n{}".format(str(unit), str(layerCount)))
                        print('\n\n\n\n\n')
                        print("##################\n{}\n{}".format(unit, opt))
                        f.write("Confusion Matrix: ")
                        f.write(str(conf_matrix))
                        f.write("Average Accuracy: {}".format(str(accuracy)))
                        f.write("Per-Class Precision: {}".format(str(precision_array)))
                        f.write("Per-Class Recall: {}".format(str(recall_array)))
                        f.write("Area under the ROC Curve: {}".format(auc))
                        tf.saved_model.save(model, "bmodels/model" + str(modelCount))
                        modelCount += 1
                    if modelCount == 50:
                        f.close()
                        sys.exit(0)
                    print(accuracy)
f.close()
# model 4 has accuracy of: 0.6844349680170576
# Model Hidden Layers: (4, 4, 4, 28)
# Hidden Layer Activations: (relu, relu, relu, relu)
# 4Confusion Matrix: [[ 108  77]
#                     [ 71 213]]
#  Average Accuracy: 0.6844349680170576
#  Per-Class Precision: [0.60335196 0.73448276]
#  Per-Class Recall: [0.58378378 0.75]
# Area under the ROC Curve: 0.6668918918918919
