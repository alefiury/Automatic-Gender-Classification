from utils.config import Config

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def construct_dataframe(genre, data_class, label):
    filelist = os.listdir(os.path.join(Config.base_dir, 'data', data_class, genre)) 

    df = pd.DataFrame(filelist)

    df['label']=label
    df = df.rename(columns={0:'file'})
    df[df['file']=='.DS_Store']

    return df

def concatenate_dfs(data_class):
    df_male = construct_dataframe('male', data_class,  '1')
    df_female = construct_dataframe('female', data_class, '0')

    df = pd.concat([df_female, df_male], ignore_index=True)

    return df

def data_standarlization(X_train, X_val, X_test):
    ss = StandardScaler()

    X_train = ss.fit_transform(X_train)
    X_train = ss.transform(X_val)
    X_test = ss.transform(X_test)

    return X_train, X_train, X_test

def data_normalization(X_train, X_val, X_test):
    
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)
    X_test = scaler.fit_transform(X_test)

    return X_train, X_train, X_test

