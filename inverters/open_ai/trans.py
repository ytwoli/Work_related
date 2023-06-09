import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
csv_file = os.getenv("ORIGIN")
db = pd.read_csv(csv_file)
goal = os.getenv("data_path")
print(db["serial"])
db["ts_day"] = pd.to_datetime(db["ts_day"], format="%Y-%m-%d %H:%M:%S.%f")
db["m.ts"] = pd.to_datetime(db["m.ts"], format="%Y-%m-%d %H:%M:%S.%f")

pd.DataFrame.to_parquet(db, path=goal)

