import time
from pickle import dump, load

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, accuracy_score, recall_score, precision_score, roc_auc_score, f1_score, \
    confusion_matrix

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from tensorflow.python.keras.models import load_model


def GridCV(model,parameter,train,train_labels,scaler):


    X_scaled =scaler.transform(train)

    gs = GridSearchCV(model, parameter, verbose=1, cv=5, n_jobs=-1,scoring='accuracy')
    gs.fit( X_scaled, train_labels)
    best_grid = gs.best_estimator_
    print(f'Best Score - {model}:', gs.best_score_)
    print(f'Best Parameter - {model}:', best_grid)
    t0 = time.time()
    mdl = best_grid.fit(train,train_labels.ravel())
    train_time = time.time() - t0
    print(f'{model} Training time :', train_time)
    return mdl
def Dimension_reduction():
    train = pd.read_csv("Data/train_sup.csv")  # read train data
    y_train = train['label']  # target
    train.drop(['label', 'type'], axis=1, inplace=True)

    test = pd.read_csv("Data/test_sup.csv")  # read test data
    y_test = test['label'] # ground truth
    test.drop(['label', 'type'], axis=1, inplace=True)

    scaler = load(open('Models/scalerencodedsup.pkl', 'rb'))
    df= scaler.transform(train)
    df= df.astype(np.float32)
    df1 = scaler.transform(test)
    df1 = df1.astype(np.float32)
    encoder = load_model('Models/encodedsup.h5')
    # encode the train data
    X_train_encode = pd.DataFrame(encoder.predict(df))
    X_train_encode =  X_train_encode.add_prefix('feature_')
    #X_train_encode = encoder.predict(df)
    X_train_encode['label'] = y_train.astype(np.float32)
    print( X_train_encode)
    # encode the test data
    X_test_encode = pd.DataFrame(encoder.predict(df1))
    X_test_encode = X_test_encode.add_prefix('feature_')
    #X_test_encode = encoder.predict(df1)
    X_test_encode['label'] = y_test.astype(np.float32)
    print(X_test_encode)
    # encode the val data


    #     X_val_encode = encoder.predict(df2)

    X_train_encode.to_csv('Data/trainSup_encoded.csv', index=False)
    X_test_encode.to_csv('Data/testSup_encoded.csv', index=False)


def train_testSupervised(model):

    train = pd.read_csv("Data/train_sup.csv") # read train data
    y_train = train['label'] # target
    train.drop(['label', 'type'], axis=1, inplace=True)

    test = pd.read_csv("Data/test_sup.csv")  # read test data
    y_test = test['label'].tolist()  # ground truth
    test.drop(['label', 'type'], axis=1, inplace=True)

    scaler = load(open('Models/scaler_sup.pkl', 'rb')) # read scaler
    test_df = scaler.transform(test)
    X_scaled = scaler.transform(train)
    t0 = time.time()
    model.fit( X_scaled,y_train)     # train model
    train_time = time.time() - t0
    print(f'{model} Training time :', train_time)

    print(' Testing ')

    t0 = time.time()
    y_pred = model.predict(test_df)
    test_time = time.time() - t0
    print(f'{model} Test time :', test_time)

    # Return performance
    conf_matrix = confusion_matrix(y_test, y_pred)

    TN = conf_matrix[0][0]
    FN = conf_matrix[1][0]
    TP = conf_matrix[1][1]
    FP = conf_matrix[0][1]
    False_positive_rate = FP / (FP + TN)
    False_negative_Rate = FN / (FN + TP)
    print(f'test_Accuracy :', accuracy_score(y_test, y_pred))
    print(f'Recall :', recall_score(y_test, y_pred))
    print(f' Precision :', precision_score(y_test, y_pred))
    print(f'  Auc_Score :', roc_auc_score(y_test, y_pred))
    print(f'   F1_score :', f1_score(y_test, y_pred))
    print('FNR:', False_negative_Rate)
    print('FPR:', False_positive_rate)

    return model
def train_testSupervised_encoded(model):

    train = pd.read_csv("Data/trainSup_encoded.csv") # read train data
    y_train = train['label'] # target
    train.drop(['label'], axis=1, inplace=True)

    test = pd.read_csv("Data/testSup_encoded.csv")  # read test data
    y_test = test['label'].tolist()  # ground truth
    test.drop(['label'], axis=1, inplace=True)


    t0 = time.time()
    model.fit( train,y_train)     # train model
    train_time = time.time() - t0
    print(f'{model} Training time :', train_time)

    print('testing ')

    t0 = time.time()
    y_pred = model.predict( test)
    test_time = time.time() - t0
    print(f'{model} Test time :', test_time)
    # Return performance
    conf_matrix = confusion_matrix(y_test, y_pred)

    TN = conf_matrix[0][0]
    FN = conf_matrix[1][0]
    TP = conf_matrix[1][1]
    FP = conf_matrix[0][1]
    False_positive_rate = FP / (FP + TN)
    False_negative_Rate = FN / (FN + TP)
    print(f'test_Accuracy :', accuracy_score(y_test, y_pred))
    print(f'Recall :', recall_score(y_test, y_pred))
    print(f' Precision :', precision_score(y_test, y_pred))
    print(f'  Auc_Score :', roc_auc_score(y_test, y_pred))
    print(f'   F1_score :', f1_score(y_test, y_pred))
    print('FNR:', False_negative_Rate)
    print('FPR:', False_positive_rate)

    return model


def split( data):
        train, test = train_test_split(data, train_size=0.8, stratify=data['type'])
        train.to_csv('Data/train_sup.csv', index=False)
        test.to_csv('Data/test_sup.csv', index=False)
def fit_sacler (data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    dump(scaler, open('Models/scaler_sup.pkl', 'wb'))

if __name__ == '__main__':
    #    Split data for training and testing
    #    df = pd.read_csv("Data/Clean.csv")
    #    split(df)
    #TO GET DIMENTION REDUCTION FOR SUPERVISED TRAINING DATA UNCOMMENT BELOW
    #    Dimension_reduction()


    LR= LogisticRegression(random_state=1, C=0.2)
    KNN = KNeighborsClassifier(metric='manhattan', weights='distance', n_neighbors=13)
    RF = RandomForestClassifier(max_depth=8, max_features='auto', n_estimators=500,
                                random_state=1)
    SVM= SVC(random_state=1, C=0.01, kernel='poly')
    print('Train and test supervised models without dimension reduced')
    train_testSupervised(LR)
    train_testSupervised(KNN)
    train_testSupervised(RF)
    train_testSupervised(SVM)
    print('Train and test supervised models with dimension reduced')
    train_testSupervised_encoded(LR)
    train_testSupervised_encoded(KNN)
    train_testSupervised_encoded(RF)
    train_testSupervised_encoded(SVM)

    # TO OPTIMIZE THE MODELS UNCOMMENT THE SECTION BELOW
    """
    PARAMETERS OPTIMIZATION FOR EACH MODEL
    train = pd.read_csv("Data/train_sup.csv")
    fit_sacler (train)
    y = train['label']
    train.drop(['label', 'type'], axis=1, inplace=True)
    knn_grid = {'n_neighbors': [5, 7, 9, 11, 13, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['minkowski', 'euclidean', 'manhattan']}
    knn = KNeighborsClassifier()
    mdl = GridCV(model=knn, parameter=knn_grid, train=train, train_labels=y, scaler=scaler)
    
    grid = {"C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], "penalty": ['none', 'l1', 'l2', 'elasticnet'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear']}
    logreg = LogisticRegression(random_state=1)
    mdl = GridCV(model=logreg, parameter=grid, train=X_scaled, train_labels=y, scaler=scaler)
    
    RF_grid1 = {
        'n_estimators': [100, 200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }
    rf = RandomForestClassifier(random_state=1)
    mdl = GridCV(model=rf, parameter=RF_grid1, train=X_scaled, train_labels=y, scaler=scaler)
    
    svm_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [ 0.01, 0.001, 0.0001],
                  'kernel': ['rbf', 'poly', 'sigmoid']}
    svm = SVC( random_state=1)
    mdl1= GridCV(model=svm, parameter=svm_grid, train=train, train_labels=y)
     
   """