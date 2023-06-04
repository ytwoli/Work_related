from sklearn import datasets
import pandas as pd

data=pd.read_parquet("data/driver_stats.parquet")
print(data)