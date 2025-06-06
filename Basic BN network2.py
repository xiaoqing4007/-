import torch
import torch.nn as nn

class BN_Manual(nn.Module):
    def __init__(self, num_features, eps=1e-5,momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
        self.momentum = momentum
        print("Init:gamma,beta",self.gamma.shape, self.beta.shape)
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def forward(self, x):
        if self.training:
            # 计算当前批次的均值和方差
            batch_mean = x.mean([0,2,3], keepdim=True)  #[0, 2, 3] 表示在 第 0、2、3 维度 上计算均值：32，16，32，32， 变为1，16，1，1
            batch_var = x.var([0,2,3], keepdim=True, unbiased=False)
            print("training: batch_mean.shape,batch_var.shape:",batch_mean.shape,batch_var.shape)
            # 更新滑动均值和方差
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*batch_mean.squeeze()
            self.running_var = (1-self.momentum)*self.running_var + self.momentum*batch_var.squeeze()
            print("self.running_mean.shape,self.running_var.shape:",self.running_mean.shape,self.running_var.shape)#[1, 16, 1, 1]
        else:
            # 使用滑动均值和方差
            batch_mean = self.running_mean.view(1, -1, 1, 1)
            batch_var = self.running_var.view(1, -1, 1, 1)
            print("testing: batch_mean.shape,batch_var.shape:",batch_mean.shape,batch_var.shape)

        # 批量归一化,
        x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        print("x_hat:",x_hat.shape,self.gamma.shape)#[32, 16, 32, 32]) torch.Size([16])
        y = self.gamma.view(1, -1, 1, 1) * x_hat + self.beta.view(1, -1, 1, 1) #gamma 缩放， beta 平移 shape：1，16，1，1
        return y
    

# 测试
x = torch.randn(32, 16, 32, 32)
BN_Manual0 = BN_Manual(num_features=16)
# BN_Manual0.training = False
# bn_layer = BN_Manual0(num_features=16)
output = BN_Manual0(x)
print(output.shape)