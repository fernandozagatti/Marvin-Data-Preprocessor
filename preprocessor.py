#Import de pré-processamento
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, QuantileTransformer 
from sklearn.feature_selection import VarianceThreshold

from pandas import read_csv, get_dummies

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

#Remover outlier
def remove_outliers(data):

    real = [i for i in range(len(data.iloc[0])) if type(data.iloc[0, i]) != str]
    mean = data.describe().iloc[1, :]
    std = data.describe().iloc[2, :]

    for (real, mean, std) in zip(real, mean, std):
        data = data[data[real] < 3*std + mean]

    return data

dataset = load_csv('Iris.csv')
dataset = remove_outliers(dataset)
#dataset = one_hot_encoder(dataset)
#dataset = minmax(dataset)
print(dataset)