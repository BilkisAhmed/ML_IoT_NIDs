from pickle import dump

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import argmin
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
tf.random.set_seed(2017)
np.random.seed(2017)

"""
class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def data_encoding(data):
    #    dataset = data.drop(['src_ip', 'src_port', 'dst_ip', 'dst_port', 'type'], axis=1)
    sel_cols = list(data.select_dtypes(include='object'))
    data1 = MultiColumnLabelEncoder(
        columns=sel_cols
    ).fit_transform(data)
    return data1


def normalise_data(data):
    # datset = data_encoding(data)
    scaler = MinMaxScaler()
    model = scaler.fit_transform(data)

    return model





def train():
    LABELS = ['0', '1']

    #Read Training Data
    #df = pd.read_csv("Data/Train_datauns.csv")
    df = pd.read_csv("Data/train_sup.csv")
    # df1 = pd.read_csv("Data/Test_dataUns.csv")
    #df1.drop(['label', 'type'], axis=1, inplace=True)
    df.drop(['label', 'type'], axis=1, inplace=True)
    #    scaler.fit_transform(df)
    #    scaler.fit_transform(df)
    #    scaler = MinMaxScaler([-1, 1])
    print('Transforming data')
    #    scaler.fit_transform(df)
    #    data = pd.read_csv('normal_data.csv')
    #    dataset = data_encoding(data)
    #    dat = dataset.drop(columns=['label'])

    # Split Training Data
    print('Splitting data')
    x_train, x_opt, x_test = np.split(df.sample(frac=1, random_state=17),
                                      [int(1 / 3 * len(df)), int(2 / 3 * len(df))])
    #  Normalize Training Data
    scaler = MinMaxScaler([-1, 1])
    print('Transforming data')
    scaler.fit_transform(x_train.append(x_opt))
    dump(scaler, open('Models/scalerencodedsup.pkl', 'wb'))
    x_train = scaler.transform(x_train.append(x_opt))
    #    x_opt = scaler.transform(x_opt)
    x_tes = scaler.transform(x_test)
    input_dim = x_train.shape[1]

    x_train = x_train.astype(np.float32)
    #    x_opt = x_opt.astype(np.float32)
    #    x_train = x_train.append(x_opt)
    x_tes = x_tes.astype(np.float32)

    encoding_dim = int(input_dim * 0.75)
    hidden_dim_1 = int(input_dim * 0.5)  # hidden_dim_2 = int(input_dim * 0.33)
    hidden_dim_2 = int(input_dim * 0.3)
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
            #    encoder

    encoder = tf.keras.layers.Dense(encoding_dim, activation='softsign', name='encoder3')(input_layer)
    encoder = tf.keras.layers.Dense(hidden_dim_1, activation='tanh', name='encoder4')(encoder)

    latent = tf.keras.layers.Dense(int(0.25 * input_dim), activation='tanh', name='latent_encoding')(encoder)
            #    Decoder

    decoder = tf.keras.layers.Dense(hidden_dim_1, activation='tanh', name='decoder2')(latent)
    decoder = tf.keras.layers.Dense(encoding_dim, activation='tanh', name='decoder3')(decoder)

    reconstructed_data = tf.keras.layers.Dense(input_dim, activation='softsign')(decoder)

            # Autoencoder
    model = tf.keras.Model(inputs=input_layer, outputs=reconstructed_data)

    model.summary()

    cp = tf.keras.callbacks.ModelCheckpoint(filepath="Models/autoencoder_AnomalyNetsup.h5",
                                            mode='min', monitor='val_loss', verbose=1, save_best_only=True)
    # define our early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True)

    model.compile(metrics=['mse'],
                  loss='mean_squared_error',
                  optimizer='Adam')

    history = model.fit(x_train, x_train,
                            epochs=100,
                            batch_size=16,
                            shuffle=True,
                            validation_data=(x_tes,x_tes),
                            verbose=1,
                            callbacks=[cp, early_stop]
                            ).history



    best_model_accuracy = history['val_mse'][argmin(history['loss'])]
    encoded = tf.keras.Model(inputs= input_layer, outputs = latent)
    encoded.save('encodedsup.h5')
    x_opt_predictions = model.predict(x_tes)
    mse = np.mean(np.power(x_tes - x_opt_predictions, 2), axis=1)
    print("mean is %.5f" % mse.mean())
    print("min is %.5f" % mse.min())
    print("max is %.5f" % mse.max())
    print("std is %.5f" % mse.std())

    import matplotlib.pyplot as plt

    plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})


    # tr = mad_score(mse)
    # mse.mean() + mse.std()
    tr =  np.quantile(mse, .90)
    with open('tresholdNet.txt', 'w') as t:
            t.write(str(tr))
    print(f"Calculated threshold is {tr}")
    over_tr = mse > tr
    false_positives = sum(over_tr)
    test_size = mse.shape[0]
    test_size = mse.shape[0]
    print(f"{false_positives} false positives on dataset without attacks with size {test_size}")

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.legend(['loss on train data', [' loss on validation data']])
    plt.show()


 df = pd.read_csv("Data/train_sup.csv")
"""
def train_autoencoder(data):



    data.drop(['label', 'type'], axis=1, inplace=True)




    # Split Training Data
    print('Splitting data')
    x_train, x_opt, x_test = np.split(data.sample(frac=1, random_state=17),
                                      [int(1 / 3 * len(data)), int(2 / 3 * len(data))])
    #  Normalize Training Data
    scaler = MinMaxScaler([-1, 1])
    print('Transforming data')
    scaler.fit_transform(x_train.append(x_opt))
    #  Save scaler and apply to data later

    x_train = scaler.transform(x_train.append(x_opt))
    x_tes = scaler.transform(x_test)
    x_train = x_train.astype(np.float32)
    x_tes = x_tes.astype(np.float32)
    input_dim = x_train.shape[1]

    # Autoencoder Model
            #    encoder
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    encoder = tf.keras.layers.Dense(int(input_dim * 0.75), activation='softsign', name='encoder3')(input_layer)
    encoder = tf.keras.layers.Dense(int(input_dim * 0.5), activation='tanh', name='encoder4')(encoder)
    latent = tf.keras.layers.Dense(int(0.25 * input_dim), activation='tanh', name='latent_encoding')(encoder)
            #    Decoder
    decoder = tf.keras.layers.Dense(int(input_dim * 0.5), activation='tanh', name='decoder2')(latent)
    decoder = tf.keras.layers.Dense(int(input_dim * 0.75), activation='tanh', name='decoder3')(decoder)
    reconstructed_data = tf.keras.layers.Dense(input_dim, activation='softsign')(decoder)
    # Autoencoder
    model = tf.keras.Model(inputs=input_layer, outputs=reconstructed_data)
    model.summary()
    model.compile(metrics=['mse'],
                  loss='mean_squared_error',
                  optimizer='Adam')
    history = model.fit(x_train, x_train,
                            epochs=5,
                            batch_size=16,
                            shuffle=True,
                            validation_data=(x_tes,x_tes),
                            verbose=1

                            ).history

    encoded = tf.keras.Model(inputs= input_layer, outputs = latent)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.legend(['loss on train data', [' loss on validation data']])
    plt.show()

    return encoded,scaler
if __name__ == '__main__':
    #For supervised models
    df = pd.read_csv("Data/train_sup.csv")
    encoded,scaler =  train_autoencoder(data=df)
    encoded.save('Models/encodedsup.h5')
    dump(scaler, open('Models/scalerencodedsup.pkl', 'wb'))
    #For unsupervised models
    df1 = pd.read_csv("Data/Train_datauns.csv")
    encoded, scaler = train_autoencoder(data=df1)
    encoded.save('Models/Unsupervised_encoded.h5')
    dump(scaler, open('Models/scalerencoded_Unsup.pkl', 'wb'))







