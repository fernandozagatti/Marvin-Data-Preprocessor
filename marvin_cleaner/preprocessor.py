from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, QuantileTransformer, KBinsDiscretizer 
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

import numpy as np
import pandas as pd
from pandas import read_csv, get_dummies

import numpy as np

global data_columns

#Apply One-Hot-Encoder
def one_hot_encoder(data):

    data = get_dummies(data)
    global data_columns
    data_columns = list(data.columns) 
    return data

####### SCALERS ####### 

#Apply MinMax Scaler
def minmax(data):

    global data_columns
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, columns=data_columns)
    return data

#Apply Standard Scaler
def standard_scaler(data):

    global data_columns
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, columns=data_columns)
    return data

#Apply Variance Threshold
def variance_threshold(data): 

    global data_columns
    selector = VarianceThreshold()
    data = selector.fit_transform(data)
    data = pd.DataFrame(data, columns=data_columns)
    return data

#Apply Robust Scaler
def robust_scaler(data):

    global data_columns
    transformer = RobustScaler().fit(data)
    data = transformer.transform(data)
    data = pd.DataFrame(data, columns=data_columns)
    return data

#Apply Normalizer Scaler
def normalizer(data):

    global data_columns
    transformer = Normalizer().fit(data)
    data = transformer.transform(data)
    data = pd.DataFrame(data, columns=data_columns)
    return data

#Apply Quantile Transformer
def quantile_transformer(data):

    global data_columns
    transformer = QuantileTransformer()
    data = transformer.fit_transform(data)
    data = pd.DataFrame(data, columns=data_columns)
    return data

####### SCALERS #######

#Data imputation
def imputation(data):

    global data_columns
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = imp.fit_transform(data)
    data = pd.DataFrame(data, columns=data_columns)
    return data

#Data discretization
def discretization(data):
    est = KBinsDiscretizer()
    data = est.fit_transform(data)

    return data

#Remove outliers of data
def remove_outlier(df_in):
    df_out = df_in
    for col_name in df_out.columns:
        q1 = df_out[col_name].quantile(0.25) #return fist quartile
        q3 = df_out[col_name].quantile(0.75) #return third quartile
        iqr = q3-q1                          #calculate interquartile range
        fence_low  = q1-1.5*iqr              #calculates the minimum value to apply to the filter
        fence_high = q3+1.5*iqr              #calculates the maximum value to apply to the filter
        df_out = df_out.loc[(df_out[col_name] > fence_low) & (df_out[col_name] < fence_high)]
    return df_out 