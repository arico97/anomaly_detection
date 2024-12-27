import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame, imbalaced_class = 'Class', test_size = 0.2):
    train, test = train_test_split(df, 
                                   test_size= test_size, 
                                   stratify=df[imbalaced_class], 
                                   random_state=42)
    return train, test