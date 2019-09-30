from MarvinCleaner import MarvinCleaner
import pandas as pd

#Test of preprocess 
'''
data = MarvinCleaner()
data.load_csv('Iris.csv')
data.preprocess(pipeline=["one_hot_encoder", "imputation", "remove_outlier", "minmax"])
data.print_dataframe()
dataframe = data.get_dataframe()
'''

#Data preparation with NLP. 
'''
data = MarvinCleaner()
data.load_csv('movie_review.csv')
data.preprocess_text(pipeline=["tokenizer","stop_words"])
print(data.dataframe)
'''

#Test with imported dataset
'''
dataframe = pd.read_csv('Iris.csv')
dataframe.drop(columns=['Species','Id'], inplace=True)
data = MarvinCleaner()
data.set_dataframe(dataframe)
data.preprocess(pipeline=["imputation", "remove_outlier", "minmax"])
data.print_dataframe()
'''