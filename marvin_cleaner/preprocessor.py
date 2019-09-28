#Import de pré-processamento
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, QuantileTransformer, KBinsDiscretizer 
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

import numpy as np
import pandas as pd
from pandas import read_csv, get_dummies

import numpy as np

#Carregar o CSV
def load_csv(filePath, missing_headers=False):

    if missing_headers:
        data = read_csv(filePath, header=None)
    else:
        data = read_csv(filePath, header=0)

    return data

#Aplicar One-Hot-Encoder
def one_hot_encoder(data):

    data = get_dummies(data)
    
    return data

#MinMax Scaler
def minmax(data):

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    return data

#Standard Scaler
def standard(data):

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    return data

#Variance Threshold
def threshold(data):

    selector = VarianceThreshold()
    data = selector.fit_transform(data)
    
    return data

#Robust Scaler
def robust(data):

    transformer = RobustScaler().fit(data)
    data = transformer.transform(data)
    
    return data

#Normalizer Scaler
def normalizer(data):

    transformer = Normalizer().fit(data)
    data = transformer.transform(data)
    
    return data

#Quantile Transformer
def quantile_transformer(data):

    transformer = QuantileTransformer()
    data = transformer.fit_transform(data)
    
    return data

#Imputação
def imputation(data):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = imp.fit_transform(data)

    return data

#Discretização
def discretization(data):
    est = KBinsDiscretizer()
    data = est.fit_transform(data)

    return data

#Removendo outlier
def remove_outlier(df_in):
    df_out = df_in
    for col_name in df_out.columns:
        q1 = df_out[col_name].quantile(0.25) #retorna o primeiro quartil
        q3 = df_out[col_name].quantile(0.75) #retorna o terceiro quartil
        iqr = q3-q1                          #calcula o iqr(interquartile range)
        fence_low  = q1-1.5*iqr              #calcula o valor minimo para aplicar no filtro
        fence_high = q3+1.5*iqr              #calcula o valor máximo para aplicar no filtro
        df_out = df_out.loc[(df_out[col_name] > fence_low) & (df_out[col_name] < fence_high)]
    return df_out 


dataset = load_csv('Iris.csv')
#dataset = one_hot_encoder(dataset)
#dataset = minmax(dataset)

dataset.drop(columns=['Species','Id'], inplace=True)
dataset = remove_outlier(dataset)

print(dataset)