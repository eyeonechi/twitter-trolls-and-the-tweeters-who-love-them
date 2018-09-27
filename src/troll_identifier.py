from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostRegressor

# ???
import lightgbm as lgb
import xgboost as xgb

import numpy as np
import pandas as pd

x_train = dataset[:,0:10]
y_train = dataset[:,10:]
seed = 1
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=seed)

# identify feature and response variables and values must be numeric and numpy arrays
# assumed have X (predictor) and Y (target) for training dataset and x_test (predictor) of test dataset
x_train = input_variables_values_training_datasets
y_train = target_variables_values_training_datasets
x_test = input_variables_values_test_datasets

''' Linear and Logistic Regression '''
def supervised_learning(x_train, y_train, x_test):
    # create model object
    model = LinearRegression()

    model = LogisticRegression()

    model = DecisionTreeClassifier(criterion='gini')
    model = DecisionTreeClassifier(criterion='entropy')
    model = DecisionTreeRegressor()

    model = SVC(gamma='scale', decision_function_shape='ovo') #kernel=
    model = LinearSVC()
    model = SVR()

    model = GaussianNB()
    model = BernoulliNB()

    model = KNeighborsClassifier(n_neighbors=6)

    model = RandomForestClassifier()

    # train the model using the training sets and check score
    model.fit(x_train, y_train)
    model.score(x_train, y_train)

    # equation coefficient and intercept
    print('coefficient:', model.coef_)
    print('intercept:', model.intercept_)

    # predict output
    predicted = model.predict(x_test)

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

def gradient_boosting(x_train, y_train, x_test):
    # create model object
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    model = XGBClassifier()

    # train the model using the training sets and check score
    model.fit(x_train, y_train)

    # predict output
    predicted = model.predict(x_test)

def lightGBM():
    data = np.random.rand(500, 10)
    label = np.random.randint(2, size=500)
    train_data = lgb.Dataset(data, label=label)
    test_data = train_data.create_valid('test.svm')
    param = {'num_leaves': 31, 'num_trees': 100, 'objective': 'binary'}
    param['metric'] = 'auc'
    num_round = 10
    bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])
    bst.save_model('model.txt')

    # 7 entities, each contains 10 features
    data = np.random.rand(7, 10)
    ypred = bst.predict(data)

def catBoostRegressor():
    #Read training and testing files
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    #Imputing missing values for both train and test
    train.fillna(-999, inplace=True)
    test.fillna(-999,inplace=True)

    #Creating a training set for modeling and validation set to check model performance
    X = train.drop(['Item_Outlet_Sales'], axis=1)
    y = train.Item_Outlet_Sales

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=1234)
    categorical_features_indices = np.where(X.dtypes != np.float)[0]

    model = CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')
    model.fit(x_train, y_train, cat_features=categorical_features_indices, eval_set=(x_train_validation, y_train_validation), plot=True)

    submission = pd.DataFrame()
    submission['Item_Identifier'] = test['Item_Identifier']
    submission['Outlet_Identifier'] = test['Outlet_Identifier']
    submission['Item_Outlet_Sales'] = model.predict(test)

def main():
    printf('hello world')
