from MarvinCleaner import MarvinCleaner

data = MarvinCleaner()
data.load_csv('Iris.csv')
data.preprocess(pipeline=["one_hot_encoder", "imputation", "remove_outlier", "minmax"])
data.print_dataframe()
dataframe = data.get_dataframe()
#data.set_dataframe(dataframe)