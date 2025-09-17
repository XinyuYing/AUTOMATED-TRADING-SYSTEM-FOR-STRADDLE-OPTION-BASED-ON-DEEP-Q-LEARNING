import pandas as pd
Data_file=["RESSET_INDXSH2019_000905.csv",
           "RESSET_INDXSH2020_000905.csv",
           "RESSET_INDXSH2021_000905.csv"]
Data=pd.read_csv("Data/15m000905/RESSET_INDXSH2018_000905.csv")
for file_name in Data_file:
    Data1=pd.read_csv("Data/15m000905/"+file_name)
    Data=pd.concat([Data,Data1],axis=0)

Data.to_csv("Data/15m000905/RESSET_INDXSH2018-2021_000905.csv",index=False)