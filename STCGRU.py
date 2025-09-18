import torch
from torch import nn
class tcnn(nn.Module):
    def __init__(self,tunnelNums):
        super(tcnn, self).__init__()
        self.num = tunnelNums
        self.largeLayer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1,
                    kernel_size=(1, 77), stride=(1,3),bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Conv2d(in_channels=1, out_channels=1,
                    kernel_size=(1, 39), stride=(1,3), bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        # 小卷积层
        self.smallLayer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1,
                    kernel_size=(1, 21), stride=(1,3), bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(in_channels=1, out_channels=1,
                    kernel_size=(1,11), stride=(1,3), bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )
    def forward(self, X):
        Y1 = self.largeLayer(X)
        Y2 = self.smallLayer(X)
        return torch.cat((Y1, Y2), dim=3)
    
class scnn(nn.Module):
    def __init__(self,tunnelNums):
        super(scnn, self).__init__()
        self.s = nn.Sequential(
            nn.Conv1d(in_channels=tunnelNums, out_channels=tunnelNums,
                kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(num_features=tunnelNums),
            nn.ReLU(),
            nn.Conv1d(in_channels=tunnelNums, out_channels=2*tunnelNums,
                kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(num_features=2*tunnelNums),
            nn.ReLU(),
            nn.Conv1d(in_channels=2*tunnelNums, out_channels=4*tunnelNums,
                    kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(num_features=4*tunnelNums),
            nn.ReLU(),
        )
    def forward(self, X):
        out = self.s(X.view(X.size(0),X.size(2),-1))
        return out


class gru(nn.Module):
    def __init__(self,tunnelNums):
        super(gru, self).__init__()
        self.g = nn.Sequential(
            nn.GRU(input_size=4*tunnelNums, hidden_size=32,
                        num_layers=1, batch_first=True, bidirectional=True)             #data_type(batch_size,timeseq,features)
        )
    def forward(self, X):
        X = X.view(X.size(0),X.size(2),-1)
        output, h_n = self.g(X)
        return h_n.permute(1,0,2)
    

    
t = tcnn(19)
s = scnn(19)
g = gru(19)
net = nn.Sequential(t,
                    s,
                    g,
                    nn.Flatten(),
                    nn.Linear(64, 2))

X = torch.rand(5,1,19,2500)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

def model(tunnelNums):
    t = tcnn(tunnelNums)
    s = scnn(tunnelNums)
    g = gru(tunnelNums)
    net = nn.Sequential(t,
                        s,
                        g,
                        nn.Flatten(),
                        nn.Linear(64, 2))
    return net
