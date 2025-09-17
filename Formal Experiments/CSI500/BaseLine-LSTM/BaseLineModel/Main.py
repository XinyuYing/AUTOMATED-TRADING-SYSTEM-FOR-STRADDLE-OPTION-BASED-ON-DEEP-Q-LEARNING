import torch
from BaseLineModel.Predict_LSTM import Predict_LSTM
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X,Y=torch.load('Data/BaseLineLSTMData/train.pt')
model=Predict_LSTM(state_dim=X.shape[-1],hidden_size=64)
model.train(X=X,Y=Y,epoch=100)
model.to(device)
torch.save(model,'ModelParm/2024-02-13LSTM.pth')
model1=torch.load('ModelParm/2024-02-13LSTM.pth')
X_test,Y_test=torch.load('Data/BaseLineLSTMData/test.pt')
X_test=X_test.to(device)

Y_test_pre=model.test(X=X_test)
