#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from preprocessor import one_hot_encoder
from preprocessor import minmax
from preprocessor import standard_scaler

class MarvinCleaner(object):
    
    def __init__(self):
        dataframe = pd.Dataframe()
        
    @classmethod
    def load_csv(cls, filePath, missing_headers=False):
        if missing_headers:
            data = pd.read_csv(file_path, header=None)
        else:
            data = pd.read_csv(filePath, header=0)
        cls.dataframe = data
    
    @classmethod
    def describe(cls):
        print(cls.dataframe.describe())
    
    PIPELINE_OPTIONS = {
        #"imputation": imputation,
        "one_hot_encoder": one_hot_encoder,
        "minmax": minmax,
        #"variance_threshold": threshold,
        #"normalizer": normalizer,
        #"quantile_transformer": quantile_transformer,
        #"robust_scaler": robust_scaler,
        "standard_scaler": standard_scaler,
    }
    
    @classmethod
    def print_dataframe(cls):
        print(cls.dataframe)

    @classmethod
    def preprocess(cls, pipeline):
        for stage in pipeline:
            if stage in cls.PIPELINE_OPTIONS:
                print("Stage --> ", stage)
                cls.dataframe = cls.PIPELINE_OPTIONS[stage](cls.dataframe)
    
        