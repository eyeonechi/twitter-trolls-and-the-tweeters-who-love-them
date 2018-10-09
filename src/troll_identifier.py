from sklearn.cluster import KMeans
from sklearn.cross_decomposition import CCA
from sklearn.cross_decomposition import PLSCanonical
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.decomposition import FactorAnalysis
# from sklearn.decomposition import PCA
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
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
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

    type = input('type: [1: supervised, 2: semi-supervised, 3: unsupervised] ')
    if type == 1:
        method = input('method: [1: classification, 2: regression] ')
        if method == 1:
            classifier = input('classifier: [1: decision tree, 2: extra tree, 3: extra trees, 4: k nearest neighbor, 5: naive bayes, 6: radius neighbors, 7: random forest, 8: support vector machine, 9: gradient boosting, 10: gaussian process, 11: stochastic gradient descent, 12: passive aggressive, 13: nearest centroid, 14: perceptron, 15: multi-layer perceptron, 16: ada boost] ')
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
                version = input('version: [1: gaussian, 2: bernoulli, 3: multinomial, 4: complement] ')
                if version == 1:
                    print(type, method, classifier, version)
                    model = GaussianNB()
                elif version == 2:
                    print(type, method, classifier, version)
                    model = BernoulliNB()
                elif version == 3:
                    print(type, method, classifier, version)
                    model = MultinomialNB()
                elif version == 4:
                    print(type, method, classifier, version)
                    model = ComplementNB()
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
            elif classifier == 9:
                print(type, method, classifier)
                model = GradientBoostingClassifier()
            elif classifier == 10:
                print(type, method, classifier)
                model = GaussianProcessClassifier(multi_class='one_vs_one')
                # model = GaussianProcessClassifier(multi_class='one_vs_rest')
            elif classifier == 11:
                print(type, method, classifier)
                model = SGDClassifier()
            elif classifier == 12:
                print(type, method, classifier)
                model = PassiveAggressiveClassifier()
            elif classifier == 13:
                print(type, method, classifier)
                model = NearestCentroid()
            elif classifier == 14:
                print(type, method, classifier)
                model = Perceptron(tol=1e-3, random_state=0)
            elif classifier == 15:
                print(type, method, classifier)
                model = MLPClassifier()
            elif classifier == 16:
                print(type, method, classifier)
                model = AdaBoostClassifier(n_estimators=100)
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

            regressor = input('regressor: [1: linear discriminant analysis, 2: logistic regression, 3: ridge regression, 4: quadratic discriminant analysis, 5: linear regression, 6: decision tree regression, 7: pls regression, 8: pls canonical, 9: canonical correlation analysis, 10: lasso, 11: multi-task lasso, 12: elastic net, 13: multi-task elastic net, 14: least angle regression, 15: least angle regression lasso, 16: orthogonal matching pursuit, 17: bayesian ridge, 18: automatic relevence determination, 19: theil sen regression, 20: huber regressor, 21: random sample consensus] ')
            if regressor == 1:
                print(type, method, regressor)
                model = LinearDiscriminantAnalysis()
            elif regressor == 2:
                print(type, method, regressor)
                model = LogisticRegression(solver='lbfgs', multi_class='multinomial') #'newton-cg'
            elif regressor == 3:
                print(type, method, regressor)
                model = RidgeClassifier()
            elif regressor == 4:
                print(type, method, regressor)
                model = QuadraticDiscriminantAnalysis()
            elif regressor == 5:
                strategy = input('strategy: [1: one vs rest, 2: one vs one] ')
                if strategy == 1:
                    print(type, method, strategy, regressor)
                    model = OneVsRestClassifier(LinearRegression())
                elif strategy == 2:
                    print(type, method, strategy, regressor)
                    model = OneVsOneClassifier(LinearRegression())
                else:
                    print('no strategy selected')
                    exit()
            elif regressor == 6:
                strategy = input('strategy: [1: one vs rest, 2: one vs one] ')
                if strategy == 1:
                    print(type, method, strategy, regressor)
                    model = OneVsRestClassifier(DecisionTreeRegressor())
                elif strategy == 2:
                    print(type, method, strategy, regressor)
                    model = OneVsOneClassifier(DecisionTreeRegressor())
                else:
                    print('no strategy selected')
                    exit()
            elif regressor == 7:
                print(type, method, regressor)
                model = PLSRegression(n_components=2)
            elif regressor == 8:
                print(type, method, regressor)
                model = PLSCanonical(n_components=2)
            elif regressor == 9:
                print(type, method, regressor)
                model = CCA(n_components=1)
            elif regressor == 10:
                print(type, method, regressor)
                model = Lasso(alpha = 0.1)
            elif regressor == 11:
                print(type, method, regressor)
                model = MultiTaskLasso(alpha=0.1)
            elif regressor == 12:
                print(type, method, regressor)
                model = ElasticNet(random_state=0)
            elif regressor == 13:
                print(type, method, regressor)
                model = MultiTaskElasticNet(random_state=0)
            elif regressor == 14:
                print(type, method, regressor)
                model = Lars(n_nonzero_coefs=1)
            elif regressor == 15:
                print(type, method, regressor)
                model = LassoLars(alpha=.1)
            elif regressor == 16:
                print(type, method, regressor)
                model = OrthogonalMatchingPursuit()
            elif regressor == 17:
                print(type, method, regressor)
                model = BayesianRidge()
            elif regressor == 18:
                print(type, method, regressor)
                model = ARDRegression()
            elif regressor == 19:
                print(type, method, regressor)
                model = TheilSenRegressor(random_state=0)
            elif regressor == 20:
                print(type, method, regressor)
                model = HuberRegressor()
            elif regressor == 21:
                print(type, method, regressor)
                model = RANSACRegressor(random_state=0)
            else:
                print('no regressor chosen')
                exit()

            # train the model using the training sets and check score
            model.fit(x_train, y_train)
            model.score(x_train, y_train)

            # print('coefficient:', model.coef_)
            # print('intercept:', model.intercept_)

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
        classifier = input('classifier: [1: label propagation, 2: label spreading] ')
        if classifier == 1:
            print(type, classifier)
            model = LabelPropagation()
        elif classifier == 2:
            print(type, classifier)
            model = LabelSpreading()
        else:
            print('no classifier chosen')
            exit()
        # train the model using the training sets and check score
        model.fit(x_train, y_train)
        model.score(x_train, y_train)

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
    elif type == 3:
        method = input('method: [1: clustering, 2: random trees embedding, 3: nearest neighbors] ')
        if method == 1:
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
            # train the model using the training sets and check score
            model.fit(x_train)

            # predict output
            predictions = model.predict(x_test)
            print('{:10}\t{:10}\t{:10}'.format('actual', 'predict', 'match?'))

            # check details
            print('centroids: ' + model.cluster_centers_)
            # print('labels: ' + model.labels_)
        elif method == 2:
            model = RandomTreesEmbedding()
            # train the model using the training sets and check score
            model.fit(x_train)

            # predict output
            predictions = model.apply(x_test)
            print('{:10}\t{:10}\t{:10}'.format('actual', 'predict', 'match?'))
        elif method == 3:
            model = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
            # train the model using the training sets and check score
            model.fit(x_train)
            distances, indices = nbrs.kneighbors(X)

        else:
            print('no method chosen')
            exit()

        # calculate accuracy
        numerator = 0.0
        denominator = float(len(predictions))
        for i in range(len(predictions)):
            match = True if (y_test[i] == predictions[i]) else False
            numerator += 1 if match else 0
            print('{:10}\t{:10}\t{:10}'.format(y_train[i], predictions[i], match))
        print('accuracy = {:7.2f}%'.format(100 * numerator / denominator))
    else:
        print('no type chosen')
        exit()



if __name__ == "__main__":
    main()
