from torch import nn
import torch.nn.functional as F
from Setting import arg
import torch
class Critic_LSTM(nn.Module):
    def __init__(self, state_dim, hidden_size, num_layer=2):
        super(Critic_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_size, num_layers=num_layer,batch_first=True)
        self.l2=nn.Linear(hidden_size,hidden_size)
        self.l3 = nn.Linear(hidden_size, 3)

    def forward(self, x):#要加入阻力位和持有时间信息
        out, (h_n, c_n) = self.lstm(x)
        x1 = h_n
        x2 = x1[-1,:,:]

        x3 = self.l2(x2)
        x3=F.leaky_relu(x3)
        x4=self.l3(x3)
        #x5=F.softmax(x4,dim=1)
        return x4


class Critic_Transformer(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super(Critic_Transformer, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=state_dim, nhead=1, dim_feedforward=64, batch_first=True)
        self.l2 = nn.Linear(state_dim * arg.history_data_len * arg.ADayTime, hidden_size)
        self.l3=nn.Linear(hidden_size+2,hidden_size)
        self.l4 = nn.Linear(hidden_size , 3)

    def forward(self, x,ResistancePointFlag,HoldTime):
        h_n = self.encoder(x)
        x1 = F.leaky_relu(h_n, negative_slope=0.1)
        x1 = torch.flatten(input=x1, start_dim=1, end_dim=2)
        x2 = F.leaky_relu(self.l2(x1))
        x2 = torch.cat([x2, ResistancePointFlag, HoldTime], dim=1)
        x3 = F.leaky_relu(self.l3(x2))
        x4=self.l4(x3)
        return x4