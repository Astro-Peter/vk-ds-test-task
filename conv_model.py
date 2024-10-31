from torch import nn
import torch


class ConvModel(nn.Module):
    def __init__(self, hidden_size):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.7)
        self.fc1 = nn.Linear(64 * 12, hidden_size)
        self.batch1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.batch2 = nn.BatchNorm1d(hidden_size) 
        self.fc3 = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(self.batch1(torch.relu(self.fc1(x))))
        x = self.dropout(self.batch2(torch.relu(self.fc2(x))))
        x = self.fc3(x)
        
        return x