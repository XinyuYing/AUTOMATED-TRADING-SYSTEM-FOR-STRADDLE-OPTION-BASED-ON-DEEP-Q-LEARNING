import pandas as pd
Data_file=["RESSET_INDXSH2019_000016.csv",
           "RESSET_INDXSH2020_000016.csv",
           "RESSET_INDXSH2021_000016.csv"]
Data=pd.read_csv("Data/15m000016/RESSET_INDXSH2018_000016.csv")
for file_name in Data_file:
    Data1=pd.read_csv("Data/15m000016/"+file_name)
    Data=pd.concat([Data,Data1],axis=0)

Data.to_csv("Data/15m000016/RESSET_INDXSH2018-2021_000016.csv",index=False)