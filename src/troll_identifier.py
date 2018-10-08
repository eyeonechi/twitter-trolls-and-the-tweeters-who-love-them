from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.svm import SVR
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.decomposition import PCA
# from sklearn.decomposition import FactorAnalysis

# not working with categorical data
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


import codecs as cd
import numpy as np
import pandas as pd

'''
x_train = dataset[:,0:10]
y_train = dataset[:,10:]
seed = 1
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=seed)

# identify feature and response variables and values must be numeric and numpy arrays
# assumed have X (predictor) and Y (target) for training dataset and x_test (predictor) of test dataset
x_train = input_variables_values_training_datasets
y_train = target_variables_values_training_datasets
x_test = input_variables_values_test_datasets
'''

def supervised_learning(x_train, y_train, x_test, method):
    # create model object
    # model = DecisionTreeRegressor()

    # model = LogisticRegression()
    # model = SVC(gamma='scale', decision_function_shape='ovo') #kernel=
    # model = LinearSVC()
    # model = SVR()



    # equation coefficient and intercept
    # print('coefficient:', model.coef_)
    # print('intercept:', model.intercept_)


    return predicted

'''
# assumed have X (attributes) for training dataset and x_test (attributes) of test dataset
def unsupervised_learning(x_train, x_test):
    # create model object
    model = KMeans(n_clusters=3, random_state=0)

    # train the model using the training sets and check score
    model.fit(X)

    # predict output
    predicted = model.predict(x_test)

def dimensionality_reduction(x_train, x_test):
    # create model object
    k = min(n_sample, n_features)
    model = PCA(n_components=k)
    model = FactorAnalysis()

    # reduced the dimension of training dataset
    train_reduced = model.fit_transform(x_train)

    # reduced the dimension of test dataset
    test_reduced = model.transform(x_test)
'''

train_dataset = '../data/small-csv/train-most10.csv'
test_dataset = '../data/small-csv/dev-most10.csv'

def main():

    # set up dataset
    data_train = pd.read_csv(train_dataset, names=['tweet-id', 'user-id', 'a', 'and', 'for', 'i', 'in', 'is', 'of', 'the', 'to', 'you', 'class'])
    data_test = pd.read_csv(test_dataset, names=['tweet-id', 'user-id', 'a', 'and', 'for', 'i', 'in', 'is', 'of', 'the', 'to', 'you', 'class'])
    x_train = data_train.drop(['tweet-id', 'user-id', 'class'], axis=1).apply(pd.to_numeric, errors='ignore')
    y_train = pd.Series(data_train['class'])
    x_test = data_test.drop(['tweet-id', 'user-id', 'class'], axis=1).apply(pd.to_numeric, errors='ignore')
    y_test = pd.Series(data_test['class'])
    # y_train = pd.Series([1, 3, 5, 6, 10])
    # predictions = pd.Series([1, 2, 5, 3, 8])

    type = input('type: [1: supervised, 2: unsupervised] ')
    if type == 1:
        method = input('method: [1: classification, 2: regression] ')
        if method == 1:
            classifier = input('classifier: [1: decision tree, 2: extra tree, 3: extra trees, 4: k nearest neighbor, 5: naive bayes, 6: radius neighbors, 7: random forest, 8: support vector machine] ')
            if classifier == 1:
                criterion = input('criterion: [1: gini, 2: entropy] ')
                if criterion == 1:
                    print(type, method, classifier, criterion)
                    model = DecisionTreeClassifier(criterion='gini')
                elif criterion == 2:
                    print(type, method, classifier, criterion)
                    model = DecisionTreeClassifier(criterion='entropy')
                else:
                    print('no criterion chosen')
                    exit()
            elif classifier == 2:
                print(type, method, classifier)
                model = ExtraTreeClassifier()
            elif classifier == 3:
                print(type, method, classifier)
                model = ExtraTreesClassifier()
            elif classifier == 4:
                n = input('n: [1: 1, 2: 3: 3: 5] ')
                if n == 1:
                    print(type, method, classifier, n)
                    model = KNeighborsClassifier(n_neighbors=1)
                elif n == 2:
                    print(type, method, classifier, n)
                    model = KNeighborsClassifier(n_neighbors=3)
                elif n == 3:
                    print(type, method, classifier, n)
                    model = KNeighborsClassifier(n_neighbors=5)
                else:
                    print('no n chosen')
                    exit()
            elif classifier == 5:
                version = input('version: [1: gaussian, 2: bernoulli] ')
                if version == 1:
                    print(type, method, classifier, version)
                    model = GaussianNB()
                elif version == 2:
                    print(type, method, classifier, version)
                    model = BernoulliNB()
                else:
                    print('no version chosen')
                    exit()
            elif classifier == 6:
                print(type, method, classifier)
                model = RadiusNeighborsClassifier(radius=1.0)
            elif classifier == 7:
                print(type, method, classifier)
                model = RandomForestClassifier()
            elif classifier == 8:
                print(type, method, classifier)
                model = LinearSVC(multi_class='crammer_singer') #multi_class='ovr'
            else:
                print('no classifier chosen')
                exit()
            # train the model using the training sets and check score
            model.fit(x_train, y_train)
            model.score(x_train, y_train)

            # predict output
            predictions = pd.Series(model.predict(x_test))
            print('{:10}\t{:10}\t{:10}\t{:10}'.format('actual', 'predict', 'approximate', 'match?'))

            # calculate accuracy
            numerator = 0.0
            denominator = float(len(predictions))
            for i in range(len(predictions)):
                match = True if (y_test[i] == predictions[i]) else False
                numerator += 1 if match else 0
                print('{:10}\t{:10}\t{:10}'.format(y_train[i], predictions[i], match))
            print('accuracy = {:7.2f}%'.format(100 * numerator / denominator))
        elif method == 2:
            # transform into binary classification problem
            # y_train = y_train.apply(lambda x: 0 if x == 'Other' else 1)
            # y_test = y_test.apply(lambda x: 0 if x == 'Other' else 1)

            # transform string labels into integers
            # le = LabelEncoder()
            # le.fit(y_train) # print(le.transform(['LeftTroll', 'Other', 'Other', 'RightTroll'])), print(le.inverse_transform([0, 1, 2, 1]))
            # print(le.classes_)
            #
            # y_train = le.transform(y_train)
            # y_test = le.transform(y_test)

            regressor = input('regressor: [1: linear discriminant analysis, 2: logistic regression, 3: ridge regression] ')
            if regressor == 1:
                print(type, method, regressor)
                model = LinearDiscriminantAnalysis()
            elif regressor == 2:
                print(type, method, regressor)
                model = LogisticRegression(solver='lbfgs', multi_class='multinomial') #'newton-cg'
            elif regressor == 3:
                print(type, method, regressor)
                model = RidgeClassifier()
            else:
                print('no regressor chosen')
                exit()

            # train the model using the training sets and check score
            model.fit(x_train, y_train)
            model.score(x_train, y_train)

            print('coefficient:', model.coef_)
            print('intercept:', model.intercept_)

            # predict output
            predictions = pd.Series(model.predict(x_test))
            print('{:10}\t{:10}\t{:10}'.format('actual', 'predict', 'match?'))

            # calculate accuracy
            numerator = 0.0
            denominator = float(len(predictions))
            for i in range(len(predictions)):
                match = True if (y_test[i] == predictions[i]) else False
                numerator += 1 if match else 0
                print('{:10}\t{:10}\t{:10}'.format(y_train[i], predictions[i], match))
            print('accuracy = {:7.2f}%'.format(100 * numerator / denominator))

        else:
            print('no method chosen')
            exit()
    elif type == 2:
        method = input('method: [1: clustering] ')
        if (method == 1):
            clusterer = input('clustere: [1: k means]')
            if clusterer == 1:
                clusters = input('clusters: [1: 1, 2: 2, 3: 3] ')
                if clusters == 1:
                    print(type, method, clusters)
                    model = KMeans(n_clusters=1, random_state=0)
                elif clusters == 2:
                    print(type, method, clusters)
                    model = KMeans(n_clusters=2, random_state=0)
                elif clusters == 3:
                    print(type, method, clusters)
                    model = KMeans(n_clusters=3, random_state=0)
                else:
                    print('no clusters chosen')
                    exit()
            else:
                print('no clusterer chosen')
                exit()
        else:
            print('no method chosen')
            exit()

        # train the model using the training sets and check score
        model.fit(x_train)

        # predict output
        predictions = model.predict(x_test)
        # print('{:10}\t{:10}\t{:10}'.format('actual', 'predict', 'match?'))

        # check details
        print('centroids: ' + model.cluster_centers_)
        # print('labels: ' + model.labels_)

        # calculate accuracy
        numerator = 0.0
        denominator = float(len(predictions))
        for i in range(len(predictions)):
            match = True if (y_test[i] == predictions[i]) else False
            numerator += 1 if match else 0
            # print('{:10}\t{:10}\t{:10}'.format(y_train[i], predictions[i], match))
        print('accuracy = {:7.2f}%'.format(100 * numerator / denominator))
    else:
        print('no type chosen')
        exit()



if __name__ == "__main__":
    main()
