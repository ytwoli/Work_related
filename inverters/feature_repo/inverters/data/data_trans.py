import pandas as pd
db = pd.read_csv("/Users/ooolivia/Desktop/FEAST/inverters/feature_repo/inverters/data/inverters.csv")
db["ts_day"] = pd.to_datetime(db["ts_day"], format="%Y-%m-%d %H:%M:%S.%f")

pd.DataFrame.to_parquet(db, path="/Users/ooolivia/Desktop/FEAST/inverters/feature_repo/inverters/data/inventers.parquet")


# data = pd.read_parquet(db)
# print(data)