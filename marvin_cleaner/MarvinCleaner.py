#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from preprocessor import one_hot_encoder
from preprocessor import minmax, standard_scaler, normalizer, quantile_transformer, robust_scaler 
from preprocessor import imputation
from preprocessor import variance_threshold
from preprocessor import remove_outlier

class MarvinCleaner(object):
    
    def __init__(self):
        dataframe = pd.DataFrame()
        
    @classmethod
    def load_csv(cls, file_path, missing_headers=False):
        if missing_headers:
            data = pd.read_csv(file_path, header=None)
        else:
            data = pd.read_csv(file_path, header=0)
        cls.dataframe = data

    @classmethod
    def describe(cls):
        print(cls.dataframe.describe())
    
    PIPELINE_OPTIONS = {
        "imputation": imputation,
        "one_hot_encoder": one_hot_encoder,
        "minmax": minmax,
        "variance_threshold": variance_threshold,
        "normalizer": normalizer,
        "quantile_transformer": quantile_transformer,
        "robust_scaler": robust_scaler,
        "standard_scaler": standard_scaler,
        "remove_outlier": remove_outlier
    }
    
    @classmethod
    def print_dataframe(cls):
        print(cls.dataframe)
    
    @classmethod
    def get_dataframe(cls):
        return cls.dataframe
    
    @classmethod 
    def set_dataframe(cls, data):
        cls.dataframe = data

    @classmethod
    def preprocess(cls, pipeline):
        for stage in pipeline:
            if stage in cls.PIPELINE_OPTIONS:
                print("Stage --> ", stage)
                cls.dataframe = cls.PIPELINE_OPTIONS[stage](cls.dataframe)