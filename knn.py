from math import sqrt
import pandas as pd
from numpy import asarray
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import statistics


def get_binary_class(y):
    return 1 if y == "Romania" else 0


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


def get_data():
    # load the dataset
    dataset = pd.read_csv('honey_sample.csv')
    # split into input (X) and output (y) variables
    X_data = dataset.loc[:, 'Li':'C13protein']
    y_data = dataset['country']

    scaler = MinMaxScaler()
    scaler.fit(X_data)
    X = scaler.transform(X_data)

    Y = asarray([get_binary_class(y_i) for y_i in y_data])

    Y = asarray([Y])

    data = np.concatenate((X, Y.T), axis=1)

    return data


def get_reduced_data():
    # load the dataset
    dataset = pd.read_csv('honey_sample.csv')
    # split into input (X) and output (y) variables
    X_data = dataset.loc[:, ['Li', 'Na', 'Mg', 'Al', 'P', 'K', 'V', 'Mn', 'Fe', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Nb',
                             'Mo', 'Pd', 'Sb', 'Ba', 'Ce', 'Ir', 'DeltaD', 'Delta18O', 'C13', 'C13protein']]
    y_data = dataset['country']

    scaler = MinMaxScaler()
    scaler.fit(X_data)
    X = scaler.transform(X_data)

    Y = asarray([get_binary_class(y_i) for y_i in y_data])

    Y = asarray([Y])

    data = np.concatenate((X, Y.T), axis=1)

    return data


def run_knn(number_neighbors, reduced):

    if reduced == 1:
        data = get_reduced_data()
    else:
        data = get_data()

    k_fold = KFold(n_splits=10, shuffle=True)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    accuracies = []
    precisions = []
    recalls = []
    sensitivities = []
    f_measures = []
    aucs = []
    for train, test in k_fold.split(data):
        current_tp = 0
        current_tn = 0
        current_fp = 0
        current_fn = 0
        for index in test:
            prediction = predict_classification(data[train], data[index], number_neighbors)
            # print('Expected %d, Got %d.' % (data[index][-1], prediction))
            if data[index][-1] == prediction and data[index][-1] == 1:
                tp += 1
                current_tp += 1
            elif data[index][-1] == prediction and data[index][-1] == 0:
                tn += 1
                current_tn += 1
            elif data[index][-1] != prediction and data[index][-1] == 1:
                fn += 1
                current_fn += 1
            elif data[index][-1] != prediction and data[index][-1] == 0:
                fp += 1
                current_fp += 1
        current_accuracy = (current_tp + current_tn) / (current_tp + current_tn + current_fn + current_fp)
        current_sensitivity = current_tp / (current_tp + current_fn)
        current_specificity = current_tn / (current_tn + current_fp)
        current_auc = (current_sensitivity + current_specificity) / 2
        current_precision = current_tp / (current_tp + current_fp)
        current_recall = current_tp / (current_tp + current_fn)
        current_f_score_pos = 2 / ((1 / (current_tp / (current_tp + current_fp))) + (1 / (current_tp / (current_tp + current_fn))))
        current_f_score_neg = 2 / ((1 / (current_tn / (current_tn + current_fn))) + (1 / (current_tn / (current_tn + current_fp))))
        current_f_score = (current_f_score_pos + current_f_score_neg) / 2
        accuracies.append(current_accuracy)
        precisions.append(current_precision)
        recalls.append(current_recall)
        sensitivities.append(current_sensitivity)
        f_measures.append(current_f_score)
        aucs.append(current_auc)

    print(accuracies)
    print('Accuracy mean = %f' % statistics.mean(accuracies))
    print(precisions)
    print('Precision mean= %f' % statistics.mean(precisions))
    print(recalls)
    print('Recall mean= %f' % statistics.mean(recalls))
    print(sensitivities)
    print('Sensitivity mean= %f' % statistics.mean(sensitivities))
    print(f_measures)
    print('F-measure mean= %f' % statistics.mean(f_measures))
    print(aucs)
    print('AUC mean= %f' % statistics.mean(aucs))
    print(statistics.mean(accuracies))
    accuracy = (tp + tn) / (tp + tn + fn + fp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc = (sensitivity + specificity) / 2
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score_pos = 2 / ((1 / (tp / (tp + fp))) + (1 / (tp / (tp + fn))))
    f_score_neg = 2 / ((1 / (tn / (tn + fn))) + (1 / (tn / (tn + fp))))
    f_score = (f_score_pos + f_score_neg) / 2
    print('Accuracy = %f' % accuracy)
    print('Precision = %f' % precision)
    print('Recall = %f' % recall)
    print('Sensitivity = %f' % sensitivity)
    print('F-measure = %f' % f_score)
    print('AUC = %f' % auc)


