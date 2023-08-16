# Normalise Train Data
#
import time
from pickle import dump, load

import numpy as np
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

from sklearn.metrics import r2_score, accuracy_score, recall_score, precision_score, roc_auc_score, f1_score, \
    confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, ParameterGrid
from tensorflow.python.keras.models import load_model

from Data_preprocessing import *
def convert_prediction(prediction):
    converted_prd = prediction.copy()
    for i in range(len(prediction)):
        if prediction[i] == -1:
            converted_prd[i] = 1
        else:
            converted_prd[i] = 0
    return converted_prd
def parse_results(prediction, labels, print_result=1):
    TN = TP = FN = FP = 0
    for i in range(len(labels)):
        if prediction[i] == 1 and labels[i] == 1:
            TP += 1
        elif prediction[i] == 1 and labels[i] == 0:
            FP += 1
        elif prediction[i] == 0 and labels[i] == 1:
            FN += 1
        else:
            TN += 1
    accuracy = round((TP+TN)/len(labels), 5)
    False_positive_rate =round( FP / (FP + TN),5)
    #False_negative_Rate =round( FN / (FN + TP),5)
    Recall = round(TP / (TP + FN), 5)

    #Recall = recall_score(labels,prediction)
    Auc_Score = roc_auc_score(labels,prediction)
    specitivity = 1 - False_positive_rate
    F1_score= f1_score(labels,prediction)
    Precision = precision_score(labels,prediction)
    if print_result == 1:
        print("\tAccuracy:",accuracy)
        print("\tFalse_positive_rate:", False_positive_rate)
        # print("\tFalse_negative_Rate:", False_negative_Rate)
    return accuracy,Recall,Auc_Score ,specitivity,F1_score,Precision,False_positive_rate
def fit_sacler (data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    dump(scaler, open('Models/scalerUnsupervised1.pkl', 'wb'))

def optimise_model(train_data,Val_data,model,param_grid,scaler):
    train_data.drop(['label', 'type'], axis=1, inplace=True)
    y_true= Val_data['label'].tolist()
    print(y_true)
    Val_data.drop(['label', 'type'], axis=1, inplace=True)
    #    fit scaler on the training dataset scaler.transform(train_data)
    metric=[0,0,0]
    for g in ParameterGrid(param_grid):
        model.set_params( **g)
        model.fit(scaler.transform(train_data))
        pred_clf = model.predict(scaler.transform(Val_data))
        predictions = convert_prediction(pred_clf)
        accuracy, Recall, Auc_Score, specitivity, F1_score, Precision, False_positive_rate = parse_results(
            prediction=predictions.tolist(), labels=y_true, print_result=0)
        print(g, metric)
        if Auc_Score > metric[-1]:
            metric = [specitivity, Recall, Auc_Score]
            print(g, metric)

    return metric
def train(data,model,scaler):

    t0 = time.time()
    mdl = model.fit(scaler.transform(data))
    train_time = time.time() - t0
    print(f'{model} Training time :', train_time)
    return mdl
def test (data,model,scaler):
    y_true = data['label'].tolist()
    data.drop(['label', 'type'], axis=1, inplace=True)
    t0 = time.time()
    pred_clf = model.predict(scaler.transform(data))
    train_time = time.time() - t0
    print(f'{model} Test time :', train_time)
    predictions = convert_prediction(pred_clf)
    accuracy,Recall,Auc_Score ,specitivity,F1_score,Precision,False_positive_rate = parse_results(prediction=predictions.tolist(), labels=y_true, print_result=0)
    print('accuracy;',accuracy,'recal;',Recall,'auc;',Auc_Score ,'specitivity;',specitivity,'f1_score;',F1_score,'precison.',Precision,False_positive_rate)

def train_test(model):
    df = pd.read_csv("Data/Train_datauns.csv")
    df1 = pd.read_csv("Data/Test_dataUns.csv")
    df.drop(['label', 'type'], axis=1, inplace=True)
    scaler = load(open('Models/scalerUnsupervised1.pkl', 'rb'))
    print(f'training and test {model}')
    model=train(data=df, model=model, scaler=scaler)
    test(data=df1, model=model, scaler=scaler)
    return model



def train_testencoded(model):

    df = pd.read_csv("Data/train_encoded.csv")
    df1 = pd.read_csv("Data/test_encoded.csv")
    df.drop(['label'], axis=1, inplace=True)
    t0 = time.time()
    model.fit(df)
    train_time = time.time() - t0
    print(f'{model} Training time :', train_time)
    print('training and test ocsvm')

    y_true = df1['label'].tolist()
    df1.drop(['label'], axis=1, inplace=True)
    t0 = time.time()
    pred_clf = model.predict(df1)
    train_time = time.time() - t0
    print(f'{model} Test time :', train_time)
    predictions = convert_prediction(pred_clf)
    accuracy, Recall, Auc_Score, specitivity, F1_score, Precision, False_positive_rate = parse_results(
        prediction=predictions.tolist(), labels=y_true, print_result=0)
    print('accuracy;', accuracy, 'recall;', Recall, 'auc;', Auc_Score, 'specitivity;', specitivity, 'f1_score;',
          F1_score, 'precison.', Precision,'FPR;', False_positive_rate)
    return model

def Dimension_reduction():
    df = pd.read_csv("Data/Train_datauns.csv")

    target_Train = df['label']

    df1 = pd.read_csv("Data/Test_dataUns.csv")

    target_test = df1['label']
    df2 = pd.read_csv("Data/Val_dataUns.csv")

    target_Val = df2['label']
    df.drop(['label', 'type'], axis=1, inplace=True)
    df1.drop(['label', 'type'], axis=1, inplace=True)
    df2.drop(['label', 'type'], axis=1, inplace=True)
    scaler = load(open('Models/scalerencoded_Unsup.pkl', 'rb'))
    df= scaler.transform(df)
    df= df.astype(np.float32)
    df1 = scaler.transform(df1)
    df1 = df1.astype(np.float32)
    df2 = scaler.transform(df2)
    df2 = df2.astype(np.float32)
    print(target_test)
    encoder = load_model('Models/Unsupervised_encoded.h5')
    # encode the train data
    X_train_encode = pd.DataFrame(encoder.predict(df))
    X_train_encode =  X_train_encode.add_prefix('feature_')
    #X_train_encode = encoder.predict(df)
    X_train_encode['label'] = target_Train.astype(np.float32)
    print( X_train_encode)
    # encode the test data
    X_test_encode = pd.DataFrame(encoder.predict(df1))
    X_test_encode = X_test_encode.add_prefix('feature_')
    #X_test_encode = encoder.predict(df1)
    X_test_encode['label'] = target_test.astype(np.float32)
    print(X_test_encode)
    # encode the val data
    X_val_encode = pd.DataFrame(encoder.predict(df2))
    X_val_encode = X_val_encode.add_prefix('feature_')

    #     X_val_encode = encoder.predict(df2)
    X_val_encode['label'] = target_Val.astype(np.float32)
    print(X_val_encode)
    X_train_encode.to_csv('train_encoded.csv', index=False)
    X_test_encode.to_csv('test_encoded.csv', index=False)
    X_val_encode.to_csv('val_encoded.csv', index=False)

if __name__ == '__main__':
    # To convert the unsupervised dataset into a reduced dimension uncomment the function below
    # Dimension_reduction()
    # The following parameters set was returned with the optimization function
    ocsvm= svm.OneClassSVM(gamma=0.0003, nu=0.0001, kernel='poly')
    ell = EllipticEnvelope(random_state=0,contamination=0.5)
    iso= IsolationForest(random_state=0, n_estimators=15, contamination=0.5, max_features=7)
    lof = LocalOutlierFactor(n_neighbors=25, algorithm='kd_tree', leaf_size=15, contamination='auto', novelty=True)

    # Unsupervised Training and testing with reduced dimension
    print('Unsupervised Training and testing with reduced dimension')
    model = train_testencoded(model=lof)
    dump(model, open(f'Models/lof_encoded.ml', 'wb'))
    model = train_testencoded(model=iso)
    dump(model, open(f'Models/iso_encoded.ml', 'wb'))
    model = train_testencoded(model=ell)
    dump(model, open(f'Models/ell_encoded.ml', 'wb'))
    model = train_testencoded(model=ocsvm)
    dump(model, open(f'Models/ocsvm_encoded.ml', 'wb'))

    print('Unsupervised Training and testing without reduced dimension')
    model = train_test(model=lof)
    dump(model, open(f'Models/lof.ml', 'wb'))
    model = train_test(model=iso)
    dump(model, open(f'Models/iso.ml', 'wb'))
    model = train_test(model=ell)
    dump(model, open(f'Models/ell.ml', 'wb'))
    model = train_test(model=ocsvm)
    dump(model, open(f'Models/ocsvm.ml', 'wb'))

    """
    TO RUN OPTIMIZATION UNCOMMENT AND SPECIFY EACH MODEL AND PARAMETER
    df = pd.read_csv("Data/Train_datauns.csv")
    df1 = pd.read_csv("Data/Val_dataUns.csv")
    scaler = load(open('Models/scalerUnsupervised1.pkl', 'rb'))
    mdl1 = svm.OneClassSVM()
    mdl2 = EllipticEnvelope(random_state=0)
    mdl3 = IsolationForest(random_state=47)
    mdl4 = ( novelty = True)
    param_grid1 = { 'gamma' :[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1],'nu': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5],
    'kernel' :["linear", "poly", "rbf", "sigmoid"]}
    param_grid2 = {'contamination': [0.001, 0.00357, 0.00714, 0.01071, 0.01429, 0.01786,
                                          0.02143, 0.025, 0.02857, 0.03214, 0.03571, 0.03929,
                                          0.04286, 0.04643, 0.05]}
    param_grid3 = {'n_estimators': list(range(100, 800, 5)),
              'max_samples': list(range(100, 500, 5)),
              'contamination': [0.1, 0.2, 0.3, 0.4, 0.5],
              'max_features': [5,10,15]}
    param_grid4 ={'n_neighbors' :[3, 5, 10, 20, 30, 50], 'algorithm'  : [ "kd_tree"], 'leaf_size' : [3, 5, 10],
     'contamination' ["auto", 0.1]}
    
    optimise_model(train_data=df, Val_data=df1, model=mdl1, param_grid=param_grid1, scaler=scaler)
    
   
   """