from torch import nn
import torch.nn.functional as F
from Setting import arg
import torch

from torch.utils.data import Dataset,DataLoader
class LSTMDataset(Dataset):
    def __init__(self, X,Y):
        self.X=X
        self.Y=Y
    def __getitem__(self, index):
        # 根据索引获取样本
        return self.X[index,:,:],self.Y[index]

    def __len__(self):
        # 返回数据集大小
        return len(self.X)




class Predict_LSTM(nn.Module):
    def __init__(self, state_dim, hidden_size, num_layer=2):
        super(Predict_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_size, num_layers=num_layer,batch_first=True)
        self.l2=nn.Linear(hidden_size,hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        self.loss_func = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device=self.device)


    def forward(self, x):#要加入阻力位和持有时间信息
        out, (h_n, c_n) = self.lstm(x)
        x1 = h_n
        x2 = x1[-1,:,:]
        x3 = self.l2(x2)
        x3=F.leaky_relu(x3)
        x4=self.l3(x3)
        return x4

    def train(self,X,Y,epoch=7):
        Data = LSTMDataset(X=X, Y=Y)
        train_loader = DataLoader(Data, batch_size=64)
        for i in range(epoch):
            for X, Y in train_loader:
                X=X.to(device=self.device)
                Y=Y.to(device=self.device)
                Y_pre=self.forward(X)
                loss=self.loss_func(Y,Y_pre)
                print(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def test(self,X):
        return self.forward(X)










