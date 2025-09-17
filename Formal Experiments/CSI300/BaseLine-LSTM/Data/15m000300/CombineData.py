import pandas as pd
Data_file=["RESSET_INDXSH2019_000300.csv",
           "RESSET_INDXSH2020_000300.csv",
           "RESSET_INDXSH2021_000300.csv"]
Data=pd.read_csv("Data/15m000300/RESSET_INDXSH2018_000300.csv")
for file_name in Data_file:
    Data1=pd.read_csv("Data/15m000300/"+file_name)
    Data=pd.concat([Data,Data1],axis=0)

Data.to_csv("Data/15m000300/RESSET_INDXSH2018-2021_000300.csv",index=False)