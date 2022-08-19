import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler

def is_unique(s):
    a = s # s.values (pandas<0.24)
    return (a[0] == a).all()

def load_data(filename, samples, os=False):
    df = pd.read_csv(filename)
    rows = np.random.choice(df.index.values, samples, replace=False)
    x, y = df.iloc[rows, :-1], df.iloc[rows, [-1]].values.ravel()
    if (os and  (not is_unique(y))):
        oversample = RandomOverSampler()
        x, y = oversample.fit_resample(x, y)
    return x, y