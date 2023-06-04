from sklearn import datasets
import pandas as pd

data = datasets.load_breast_cancer()
#form the data
data_df = pd.DataFrame(data=data.data, columns=data.feature_names)
data_df1 = data_df[data.feature_names[:5]]
data_df2 = data_df[data.feature_names[5:10]]
data_df3 = data_df[data.feature_names[10:17]]
data_df4 = data_df[data.feature_names[17:30]]
target_df = pd.DataFrame(data=data.target, columns=["target"])

timestamps = pd.date_range(
    end=pd.Timestamp.now(), 
    periods=len(data_df), 
    freq='D').to_frame(name="event_timestamp", index=False)
# Adding the timestamp column to each DataFrame
data_df1 = pd.concat(objs=[data_df1, timestamps], axis=1)
data_df2 = pd.concat(objs=[data_df2, timestamps], axis=1)
data_df3 = pd.concat(objs=[data_df3, timestamps], axis=1)
data_df4 = pd.concat(objs=[data_df4, timestamps], axis=1)
target_df = pd.concat(objs=[target_df, timestamps], axis=1)
patient_ids = pd.DataFrame(data=list(range(len(data_df))), columns=["patient_id"])

# Adding the patient id column to each DataFrame
data_df1 = pd.concat(objs=[data_df1, patient_ids], axis=1)
data_df2 = pd.concat(objs=[data_df2, patient_ids], axis=1)
data_df3 = pd.concat(objs=[data_df3, patient_ids], axis=1)
data_df4 = pd.concat(objs=[data_df4, patient_ids], axis=1)
target_df = pd.concat(objs=[target_df, patient_ids], axis=1)
#convert to parquet format
data_df1.to_parquet(path='data_df1.parquet')
data_df2.to_parquet(path='data_df2.parquet')
data_df3.to_parquet(path='data_df3.parquet')
data_df4.to_parquet(path='data_df4.parquet')
target_df.to_parquet(path='target_df.parquet')