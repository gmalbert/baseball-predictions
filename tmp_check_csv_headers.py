import pandas as pd
path='data_files/retrosheet/gameinfo.csv'
df = pd.read_csv(path, nrows=2)
print(df.columns.tolist())
