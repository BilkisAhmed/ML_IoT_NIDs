# This is a sample Python script.
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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
    dataset = data.drop(['src_ip', 'src_port', 'dst_ip', 'dst_port'], axis=1)
    sel_cols = list(dataset.select_dtypes(include='object'))
    data1 = MultiColumnLabelEncoder(
        columns=sel_cols
    ).fit_transform(dataset)
    return data1
def Clean_data(data):
  missing_values = ["-"]
  df = pd.read_csv(data,na_values=missing_values)
  df["http_version"] = df["http_version"].fillna(0)
  df["weird_addl"] = df["weird_addl"].fillna(0)
  df[['http_trans_depth']] = df[['http_trans_depth']].fillna(df[['http_trans_depth']].median())
  df = df.fillna("Unknown")
  df_clean = data_encoding(df)
  df_clean.to_csv('Data/Clean.csv', index=False)
  print(df_clean)
  return df_clean
# Press the green button in the gutter to run the script.
def split_data(data):
    attack_dataset = data[data.label == 1]
    normal_dataset = data[data.label == 0]
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    for train_index, test_index in sss.split(attack_dataset, attack_dataset["type"]):
        sss_train = attack_dataset.iloc[train_index]
        sss_test = attack_dataset.iloc[test_index]

    normal_test = normal_dataset.sample(n=sss_train.shape[0], random_state=17)
    test_data = pd.concat([sss_train, normal_test])
    test_data.to_csv('Data/Test_dataUns.csv', index=False)
    normal_dataset1 = normal_dataset.drop( normal_test.index)
    df_val = normal_dataset1.sample(n=sss_test.shape[0], random_state=17)
    Val_data = pd.concat([sss_test, df_val])

    Val_data.to_csv('Data/Val_dataUns.csv', index=False)
    train_dataset = normal_dataset1.drop(df_val.index)
    #print(train_dataset.shape)
    train_dataset.to_csv('Data/Train_datauns.csv', index=False)
    return train_dataset,Val_data,test_data
if __name__ == '__main__':
    #df = "Data/ToN.csv" uncomment to return clean data
    #data=Clean_data(df)

    # Split Data for unsupervised training
    df = pd.read_csv("Data/Clean.csv")
    Train,Val,Test = split_data(df)
    print(Train.shape,Val.shape,Test.shape)


