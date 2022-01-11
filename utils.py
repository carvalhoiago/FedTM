import pandas as pd

def load_data(filename):
    df = pd.read_csv(filename)
    x, y = df.iloc[:, :-1], df.iloc[:, [-1]].values.ravel()
    return x, y