# from sklearn.cluster import KMeans
from sklearn.cross_decomposition import CCA
from sklearn.cross_decomposition import PLSCanonical
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.decomposition import FactorAnalysis
# from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import TheilSenRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.multioutput import MultiOutputRegressor
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
# from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
# from sklearn.semi_supervised import LabelPropagation
# from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import LinearSVC
# from sklearn.svm import NuSVC
# from sklearn.svm import SVC
# from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeClassifier

import codecs as cd
import numpy as np
import pandas as pd
import sys

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

# def supervised_learning(x_train, y_train, x_test, method):
#     return predicted

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

#'../data/small-csv/train-most10.csv'
#'../data/small-csv/dev-most10.csv'
# train_dataset = '../data/medium-csv/train-most50.csv'
# test_dataset = '../data/medium-csv/dev-most50.csv'
train_dataset = '../data/large-csv/train-best200.csv'
test_dataset = '../data/large-csv/dev-best200.csv'

methods = {
    1 : 'classification',
    2 : 'regression'
}

classifiers = {
    1: 'decision tree',
    2: 'extra tree',
    3: 'extra trees',
    4: 'k nearest neighbor',
    5: 'naive bayes',
    6: 'radius neighbors',
    7: 'random forest',
    8: 'support vector machine',
    9: 'gradient boosting',
    10: 'gaussian process',
    11: 'stochastic gradient descent',
    12: 'passive aggressive',
    13: 'nearest centroid',
    14: 'perceptron',
    15: 'multi-layer perceptron',
    16: 'ada boost',
    17: 'dummy'
}

regressors = {
    1: 'linear discriminant analysis',
    2: 'logistic regression',
    3: 'ridge regression',
    4: 'quadratic discriminant analysis',
    5: 'linear regression',
    6: 'decision tree regression',
    7: 'lasso',
    8: 'multi-task lasso',
    9: 'elastic net',
    10: 'multi-task elastic net',
    11: 'least angle regression',
    12: 'least angle regression lasso',
    13: 'orthogonal matching pursuit',
    14: 'bayesian ridge',
    15: 'automatic relevence determination',
    16: 'theil sen regression',
    17: 'huber regressor',
    18: 'random sample consensus'
}

def main():

    # Checks for correct number of arguments
    if len(sys.argv) != 3:
        print('usage: ./troll_identifier.py [TRAIN DATASET] [TEST/DEV DATASET]')
        sys.sys.exit()

    # set up dataset
    data_train = pd.read_csv(sys.argv[1])
    data_test = pd.read_csv(sys.argv[2])

    print('train:\n{}\n'.format(sys.argv[1]))
    print('test:\n{}\n'.format(sys.argv[2]))

    if 'small' in sys.argv[1]:
        size = 'small'
    elif 'medium' in sys.argv[1]:
        size = 'medium'
    else:
        size = 'large'

    x_train = data_train.drop([data_train.columns[0], data_train.columns[1], data_train.columns[-1]], axis=1).apply(pd.to_numeric, errors='ignore')
    y_train = pd.Series(data_train.iloc[:,-1])
    x_test = data_test.drop([data_test.columns[0], data_test.columns[1], data_test.columns[-1]], axis=1).apply(pd.to_numeric, errors='ignore')
    y_test = pd.Series(data_test.iloc[:,-1])

    # type = input('type: [1: supervised, 2: semi-supervised, 3: unsupervised] ')
    # if type == 1:
    parameter = None
    method = input('select a method: {}: '.format(methods))
    if method == 1:
        classifier = input('select a classifier: {}: '.format(classifiers))
        if classifier == 1:
            parameter = input('criterion: [1: gini, 2: entropy] ')
            if parameter == 1:
                model = DecisionTreeClassifier(criterion='gini')
                parameter = 'gini'
            elif parameter == 2:
                model = DecisionTreeClassifier(criterion='entropy')
                parameter = 'entropy'
            else:
                print('no criterion chosen')
                sys.exit()
        elif classifier == 2:
            model = ExtraTreeClassifier()
        elif classifier == 3:
            model = ExtraTreesClassifier()
        elif classifier == 4:
            parameter = input('n: [1: 1, 2: 3: 3: 5] ')
            if parameter == 1:
                model = KNeighborsClassifier(n_neighbors=1)
                parameter = '1'
            elif parameter == 2:
                model = KNeighborsClassifier(n_neighbors=3)
                parameter = '3'
            elif parameter == 3:
                model = KNeighborsClassifier(n_neighbors=5)
                parameter = '5'
            else:
                print('no n chosen')
                sys.exit()
        elif classifier == 5:
            parameter = input('version: [1: gaussian, 2: bernoulli, 3: multinomial, 4: complement] ')
            if parameter == 1:
                model = GaussianNB()
                parameter = 'gaussian'
            elif parameter == 2:
                model = BernoulliNB()
                parameter = 'bernoulli'
            elif parameter == 3:
                model = MultinomialNB()
                parameter = 'multinomial'
            elif parameter == 4:
                model = ComplementNB()
                parameter = 'complement'
            else:
                print('no version chosen')
                sys.exit()
        elif classifier == 6:
            model = RadiusNeighborsClassifier(radius=1.0)
        elif classifier == 7:
            model = RandomForestClassifier(n_estimators=50, random_state=1)
        elif classifier == 8:
            model = LinearSVC(multi_class='crammer_singer') #multi_class='ovr'
        elif classifier == 9:
            model = GradientBoostingClassifier()
        elif classifier == 10:
            model = GaussianProcessClassifier(multi_class='one_vs_one')
        elif classifier == 11:
            model = SGDClassifier()
        elif classifier == 12:
            model = PassiveAggressiveClassifier()
        elif classifier == 13:
            model = NearestCentroid()
        elif classifier == 14:
            model = Perceptron(tol=1e-3, random_state=0)
        elif classifier == 15:
            model = MLPClassifier()
        elif classifier == 16:
            model = AdaBoostClassifier(n_estimators=50)
        elif classifier == 17:
            parameter = input('strategy: [1: stratified, 2: most frequent, 3: prior, 4: uniform, 5: constant] ')
            if parameter == 1:
                model = DummyClassifier(strategy='stratified')
                parameter = 'stratified'
            elif parameter == 2:
                model = DummyClassifier(strategy='most_frequent')
                parameter = 'most frequent'
            elif parameter == 3:
                model = DummyClassifier(strategy='prior')
                parameter = 'prior'
            elif parameter == 4:
                model = DummyClassifier(strategy='uniform')
                parameter = 'uniform'
            elif parameter == 5:
                model = DummyClassifier(strategy='constant')
                parameter = 'constant'
            else:
                print('no strategy selected')
                sys.exit()
        else:
            print('no classifier chosen')
            sys.exit()

        import time
        # Starts timer
        start = time.clock()

        # train the model using the training sets and check score
        model.fit(x_train, y_train)
        model.score(x_train, y_train)

        # predict output
        predictions = pd.Series(model.predict(x_test))
        report = classification_report(y_test, predictions, target_names=['RightTroll', 'LeftTroll', 'Other'])
        confusion = confusion_matrix(y_test, predictions, labels=["RightTroll", "LeftTroll", "Other"])
        if (parameter != None):
            filename = '{},{},{},{}.txt'.format(size, methods[method], classifiers[classifier], parameter)
        else:
            filename = '{},{},{}.txt'.format(size, methods[method], classifiers[classifier])

        # Prints the time taken
        end = time.clock()
        time = str(end - start)

        with open(filename, 'w') as output:
            output.write('method:\n{}\n\n'.format(methods[method]))
            output.write('classifier:\n{}\n\n'.format(classifiers[classifier]))
            output.write('accuracy:\n{:.2f}%\n\n'.format(100 * accuracy_score(y_test, predictions)))
            output.write('report:\n{}\n\n'.format(report))
            output.write('confusion:\n{}\n\n'.format(confusion))
            output.write('time:\n{}s\n\n'.format(time))
            output.write('data:\n{:10}\t{:10}\t{:10}\n'.format('actual', 'predict', 'match?'))
            for i in range(len(predictions)):
                output.write('{:10}\t{:10}\t{:10}\n'.format(y_train[i], predictions[i], y_test[i] == predictions[i]))

        print('\nmethod:\n{}\n'.format(methods[method]))
        print('classifier:\n{}\n'.format(classifiers[classifier]))
        print('accuracy:\n{:.2f}%\n'.format(100 * accuracy_score(y_test, predictions)))
        print('report:\n{}\n'.format(report))
        print('confusion:\n{}\n'.format(confusion))
        print('time: {}s\n'.format(time))

    elif method == 2:
        # transform into binary classification problem
        # y_train = y_train.apply(lambda x: 0 if x == 'Other' else 1)
        # y_test = y_test.apply(lambda x: 0 if x == 'Other' else 1)

        # transform string labels into integers
        le = LabelEncoder()
        le.fit(y_train) # print(le.transform(['LeftTroll', 'Other', 'Other', 'RightTroll'])), print(le.inverse_transform([0, 1, 2, 1]))
        print(le.classes_)

        y_train = le.transform(y_train)
        y_test = le.transform(y_test)

        regressor = input('select a regressor: {}: '.format(regressors))
        if regressor == 1:
            print(method, regressor)
            model = LinearDiscriminantAnalysis()
        elif regressor == 2:
            print(method, regressor)
            model = LogisticRegression(solver='lbfgs', multi_class='multinomial') #'newton-cg'
        elif regressor == 3:
            print(method, regressor)
            model = RidgeClassifier()
        elif regressor == 4:
            print(method, regressor)
            model = QuadraticDiscriminantAnalysis()
        elif regressor == 5:
            model = OneVsRestClassifier(LinearRegression())
        elif regressor == 6:
            model = OneVsRestClassifier(DecisionTreeRegressor())
        elif regressor == 7:
            print(method, regressor)
            model = OneVsRestClassifier(Lasso(alpha = 0.1))
        elif regressor == 8:
            print(method, regressor)
            model = OneVsRestClassifier(MultiTaskLasso(alpha=0.1))
        elif regressor == 9:
            print(method, regressor)
            model = OneVsRestClassifier(ElasticNet(random_state=0))
        elif regressor == 10:
            print(method, regressor)
            model = OneVsRestClassifier(MultiTaskElasticNet(random_state=0))
        elif regressor == 11:
            print(method, regressor)
            model = OneVsRestClassifier(Lars(n_nonzero_coefs=1))
        elif regressor == 12:
            print(method, regressor)
            model = OneVsRestClassifier(LassoLars(alpha=.1))
        elif regressor == 13:
            print(method, regressor)
            model = OneVsRestClassifier(OrthogonalMatchingPursuit())
        elif regressor == 14:
            print(method, regressor)
            model = OneVsRestClassifier(BayesianRidge())
        elif regressor == 15:
            print(method, regressor)
            model = OneVsRestClassifier(ARDRegression())
        elif regressor == 16:
            print(method, regressor)
            model = OneVsRestClassifier(TheilSenRegressor(random_state=0))
        elif regressor == 17:
            print(method, regressor)
            model = OneVsRestClassifier(HuberRegressor())
        elif regressor == 18:
            print(method, regressor)
            model = OneVsRestClassifier(RANSACRegressor(random_state=0))
        else:
            print('no regressor chosen')
            sys.exit()

        import time
        # Starts timer
        start = time.clock()

        # train the model using the training sets and check score
        model.fit(x_train, y_train)
        model.score(x_train, y_train)

        # y_train = le.inverse_transform(y_train)
        # y_test = le.inverse_transform(y_test)
        # print('coefficient:', model.coef_)
        # print('intercept:', model.intercept_)

        # predict output
        predictions = pd.Series(model.predict(x_test))
        if (parameter != None):
            filename = '{},{},{},{}.txt'.format(size, methods[method], regressors[regressor], parameter)
        else:
            filename = '{},{},{}.txt'.format(size, methods[method], regressors[regressor])

        # Prints the time taken
        end = time.clock()
        time = str(end - start)

        with open(filename, 'w') as output:
            output.write('method:\n{}\n\n'.format(methods[method]))
            output.write('regressor:\n{}\n\n'.format(regressors[regressor]))
            output.write('accuracy:\n{:.2f}%\n\n'.format(100 * accuracy_score(y_test, predictions)))
            output.write('time:\n{}s\n\n'.format(time))
            output.write('data:\n{:10}\t{:10}\t{:10}\n'.format('actual', 'predict', 'match?'))
            for i in range(len(predictions)):
                output.write('{:10}\t{:10}\t{:10}\n'.format(y_train[i], predictions[i], y_test[i] == predictions[i]))

        print('\nmethod:\n{}\n'.format(methods[method]))
        print('regressor:\n{}\n'.format(regressors[regressor]))
        print('accuracy:\n{:.2f}%\n'.format(100 * accuracy_score(y_test, predictions)))
        print('time: {}s\n'.format(time))

    else:
        print('no method chosen')
        sys.exit()
    # elif type == 2:
    #     classifier = input('classifier: [1: label propagation, 2: label spreading] ')
    #     if classifier == 1:
    #         print(classifier)
    #         model = LabelPropagation()
    #     elif classifier == 2:
    #         print(classifier)
    #         model = LabelSpreading()
    #     else:
    #         print('no classifier chosen')
    #         sys.exit()
    #     # train the model using the training sets and check score
    #     model.fit(x_train, y_train)
    #     model.score(x_train, y_train)
    #
    #     # predict output
    #     predictions = pd.Series(model.predict(x_test))
    #     print('{:10}\t{:10}\t{:10}'.format('actual', 'predict', 'match?'))
    #
    #     # calculate accuracy
    #     numerator = 0.0
    #     denominator = float(len(predictions))
    #     for i in range(len(predictions)):
    #         match = True if (y_test[i] == predictions[i]) else False
    #         numerator += 1 if match else 0
    #         print('{:10}\t{:10}\t{:10}'.format(y_train[i], predictions[i], match))
    #     print('accuracy = {:7.2f}%'.format(100 * numerator / denominator))
    # elif type == 3:
    #     method = input('method: [1: clustering, 2: random trees embedding, 3: nearest neighbors] ')
    #     if method == 1:
    #         clusterer = input('clustere: [1: k means]')
    #         if clusterer == 1:
    #             clusters = input('clusters: [1: 1, 2: 2, 3: 3] ')
    #             if clusters == 1:
    #                 print(method, clusters)
    #                 model = KMeans(n_clusters=1, random_state=0)
    #             elif clusters == 2:
    #                 print(method, clusters)
    #                 model = KMeans(n_clusters=2, random_state=0)
    #             elif clusters == 3:
    #                 print(method, clusters)
    #                 model = KMeans(n_clusters=3, random_state=0)
    #             else:
    #                 print('no clusters chosen')
    #                 sys.exit()
    #         else:
    #             print('no clusterer chosen')
    #             sys.exit()
    #         # train the model using the training sets and check score
    #         model.fit(x_train)
    #
    #         # predict output
    #         predictions = model.predict(x_test)
    #         print('{:10}\t{:10}\t{:10}'.format('actual', 'predict', 'match?'))
    #
    #         # check details
    #         print('centroids: ' + model.cluster_centers_)
    #         # print('labels: ' + model.labels_)
    #     elif method == 2:
    #         model = RandomTreesEmbedding()
    #         # train the model using the training sets and check score
    #         model.fit(x_train)
    #
    #         # predict output
    #         predictions = model.apply(x_test)
    #         print('{:10}\t{:10}\t{:10}'.format('actual', 'predict', 'match?'))
    #     elif method == 3:
    #         model = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
    #         # train the model using the training sets and check score
    #         model.fit(x_train)
    #         distances, indices = nbrs.kneighbors(X)
    #
    #     else:
    #         print('no method chosen')
    #         sys.exit()
    #
    #     # calculate accuracy
    #     numerator = 0.0
    #     denominator = float(len(predictions))
    #     for i in range(len(predictions)):
    #         match = True if (y_test[i] == predictions[i]) else False
    #         numerator += 1 if match else 0
    #         print('{:10}\t{:10}\t{:10}'.format(y_train[i], predictions[i], match))
    #     print('accuracy = {:7.2f}%'.format(100 * numerator / denominator))
    # else:
    #     print('no type chosen')
    #     sys.exit()



if __name__ == "__main__":
    main()
