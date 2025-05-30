//
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=16*32*32, out_features=10)
    
    def forward(self, x):
        print('x:',x.shape)#32,3,32,32
        x = self.conv1(x)
        print('conv1 x:',x.shape)#32,16,32,32
        x = self.bn1(x)
        print('bn1 x:',x.shape)#32,16,32,32
        x = self.relu(x)
        print('relu x',x.shape) #32,16,32,32  打印显示用：x.shape, ！！！！ x.size(0)直接用，不能打印显示
        x = x.view(x.size(0),-1) #(32,-1)
        print('view x:',x.shape) #32,16*32*32
        x = self.fc(x) 
        print('fc x',x.shape) #32,10
        return x
    
x = torch.randn(32,3,32,32) # 批次大小为32，3通道，32x32图像
model = SimpleNet()

output = model(x)
print(output.shape)
torch.randn
