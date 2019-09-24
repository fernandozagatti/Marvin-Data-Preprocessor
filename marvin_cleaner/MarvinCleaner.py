import pandas as pd

class MarvinCleaner:

    def __init__(self):
        dataframe = pd.DataFrame()

    PIPELINE_OPTIONS = {
        "imputation": imputation,
        "onehotencoding": onehotencoding,
        "minmax": minmax,
        "normalizer": normalizer,
        "quantiletransformer": quantiletransformer,
        "robustscaler": robustscaler,
        "standardscaler": standardscaler,
    }

    def load_csv(filePath, missing_headers=False):
        if missing_headers:
            data = read_csv(filePath, header=None)
        else:
            data = read_csv(filePath, header=0)

        return data

    def getData(url=False, path):
        if not url:
            self.dataframe = load_csv(path)

    def preprocess(self, pipeline):
        for stage in pipeline:
            if stage in PIPELINE_OPTIONS:
                print("Stage --> ", stage)
                PIPELINE_OPTIONS[stage]()
