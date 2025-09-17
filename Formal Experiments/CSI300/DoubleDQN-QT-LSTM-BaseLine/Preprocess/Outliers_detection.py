import numpy as np
import pandas as pd
from Code.Preprocess.ResistanceLevel1 import ResistancePoint,SupportPoint
import statistics
data=pd.read_csv('Data/A000300/new_000300.csv')
win=3
ResistancePointFrontAmplitudes=[]
ResistancePointAfterAmplitudes=[]
#压力位的振幅过滤
for i in range(1,len(ResistancePoint)):
    MaxPoint=max(data.loc[int(ResistancePoint[i,0])-win:int(ResistancePoint[i,0]),"avg"])
    MinPoint=min(data.loc[int(ResistancePoint[i,0])-win:int(ResistancePoint[i,0]),"avg"])
    FrontAmplitude=MaxPoint-MinPoint
    ResistancePointFrontAmplitudes.append(FrontAmplitude)

    MaxPoint=max(data.loc[int(ResistancePoint[i,0]):int(ResistancePoint[i,0]+win),"avg"])
    MinPoint=min(data.loc[int(ResistancePoint[i,0]):int(ResistancePoint[i,0]+win),"avg"])
    FrontAmplitude=MaxPoint-MinPoint
    ResistancePointAfterAmplitudes.append(FrontAmplitude)

i=1
StdFrontAmplitudes=statistics.stdev(ResistancePointFrontAmplitudes)
MeanFrontAmplitudes=statistics.mean(ResistancePointFrontAmplitudes)
StdAfterAmplitudes=statistics.stdev(ResistancePointAfterAmplitudes)
MeanAfterAmplitudes=statistics.mean(ResistancePointAfterAmplitudes)
while i<ResistancePoint.shape[0]:
    if ResistancePointFrontAmplitudes[i]>MeanFrontAmplitudes+3*StdFrontAmplitudes:
        if ResistancePointAfterAmplitudes[i]>MeanAfterAmplitudes+3*StdAfterAmplitudes:
            ResistancePoint=np.delete(ResistancePoint,i,axis=0)
            i=i-1

    i=i+1


SupportPointFrontAmplitudes=[]
SupportPointAfterAmplitudes=[]
#阻力位的振幅过滤
for i in range(1,len(SupportPoint)):
    MaxPoint=max(data.loc[int(SupportPoint[i,0])-win:int(SupportPoint[i,0]),"avg"])
    MinPoint=min(data.loc[int(SupportPoint[i,0])-win:int(SupportPoint[i,0]),"avg"])
    FrontAmplitude=MaxPoint-MinPoint
    SupportPointFrontAmplitudes.append(FrontAmplitude)

    MaxPoint=max(data.loc[int(SupportPoint[i,0]):int(SupportPoint[i,0]+win),"avg"])
    MinPoint=min(data.loc[int(SupportPoint[i,0]):int(SupportPoint[i,0]+win),"avg"])
    FrontAmplitude=MaxPoint-MinPoint
    SupportPointAfterAmplitudes.append(FrontAmplitude)

i=1
StdFrontAmplitudes=statistics.stdev(SupportPointFrontAmplitudes)
MeanFrontAmplitudes=statistics.mean(SupportPointFrontAmplitudes)
StdAfterAmplitudes=statistics.stdev(SupportPointAfterAmplitudes)
MeanAfterAmplitudes=statistics.mean(SupportPointAfterAmplitudes)
while i<SupportPoint.shape[0]:
    if SupportPointFrontAmplitudes[i]>MeanFrontAmplitudes+3*StdFrontAmplitudes:
        if SupportPointAfterAmplitudes[i]>MeanAfterAmplitudes+3*StdAfterAmplitudes:
            SupportPoint=np.delete(SupportPoint,i,axis=0)
            i=i-1

    i=i+1


