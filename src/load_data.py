import pandas as pd

def load_data(data_path: str, **kwargs):
    data = pd.read_csv(data_path, sep=',', on_bad_lines='skip', **kwargs )
    return data