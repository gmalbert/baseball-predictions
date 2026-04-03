import pandas as pd
df = pd.read_csv('data_files/retrosheet/allplayers.csv', nrows=3)
print("allplayers cols:", list(df.columns))

gi = pd.read_csv('data_files/retrosheet/gameinfo.csv', nrows=3)
print("gameinfo cols:", list(gi.columns))

# Check umphome values
gi2 = pd.read_csv('data_files/retrosheet/gameinfo.csv', usecols=['umphome','ump1b','ump2b','ump3b'], nrows=20)
print("sample umphome values:", gi2['umphome'].dropna().head(5).tolist())
