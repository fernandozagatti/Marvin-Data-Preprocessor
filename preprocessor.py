#Import de pré-processamento
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, QuantileTransformer, KBinsDiscretizer 
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

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

dataset = load_csv('Iris.csv')
#dataset = one_hot_encoder(dataset)
#dataset = minmax(dataset)

#dataset.drop(columns=['Species','Id'], inplace=True)
#dataset = imputation(dataset)

#dataset = discretization(dataset)

print(dataset)