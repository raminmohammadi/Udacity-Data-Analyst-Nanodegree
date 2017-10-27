from sklearn.feature_selection import SelectKBest
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score
import numpy as np

def get_k_best(data_dict, features_list, k):
    '''
    :param data_dict: input data dictionary
    :param features_list: list of features
    :param k: number of features with highest score
    :return: returns k best features selected by sklearn SelectKbest method
    '''
    data = featureFormat(data_dict, features_list)
    target, features = targetFeatureSplit(data)
    k_best = SelectKBest(k=k).fit(features, target)
    features_scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], features_scores)
    sorted_pairs = list(sorted(unsorted_pairs, key=lambda x: x[1], reverse=True))
    k_best_features = [el[0] for el in sorted_pairs][:k]
    print(k_best_features)
    return k_best_features



import matplotlib.pyplot as plt


def plotfunction(data_dict, x, y , outlier_comment = False):
    """
    :param data_dict: a data dictionary
    :param x: feature for x axis
    :param y: feature for y axis
    :return: scatter plot of x1 and x2
    """
    data = featureFormat(data_dict, ['poi',x, y])
    for point in data:
        poi = point[0]
        x1 = point[1]
        x2 = point[2]
        if poi:
            color = 'red'
        else:
            color = 'green'
        plt.scatter(x1, x2, color=color)

    plt.xlabel(x)
    plt.ylabel(y)
    if outlier_comment:
        plt.figtext(0.02, 0.02, " Data with Outlier")
    if not outlier_comment:
        plt.figtext(0.02, 0.02, " Data without Outlier")
    plt.show()



def make_data_test(model_name, data_dict, features_list, number_of_folds):
    data = featureFormat(data_dict, features_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    features = MinMaxScaler().fit_transform(features)
    precisions = []
    recalls = []
    accuracy = []
    kf=KFold(len(labels),number_of_folds,shuffle=True)
    for train_indices, test_indices in kf:
        features_train= [features[i] for i in train_indices]
        features_test= [features[i] for i in test_indices]
        labels_train=[labels[i] for i in train_indices]
        labels_test=[labels[i] for i in test_indices]
        clf = model_name.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        precisions.append(precision_score(labels_test, pred))
        recalls.append(recall_score(labels_test, pred))
        accuracy.append(accuracy_score(pred, labels_test))
    print("Accuracy of the {} model is {}: ".format(model_name,np.average(accuracy)))
    print("Precisions value {}: ".format(np.average(precisions)))
    print("Recall value {}: ".format(np.average(recalls)))
    return clf

