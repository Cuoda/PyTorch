```Python
import torch
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.0],[2.0],[3.0]])#[[],[],[]] 3*1 3行1列的矩阵。3个样本，每个样本包含1个feature。
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

class LinearModel(torch.nn.Module):#新定义一个类LinearModel继承nn.Module
  def __init__(self):#构造函数
    super(LinearModel,self).__init__()
    self.linear = torch.nn.Linear(1,1)
    #nn.Linear为pytorch中的一个类，torch.nn.Linear(in_features,out_features,bias=True)。此处实例化(调用__init__初始化)了一个nn.Linear

  def forward(self, x):
    y_pred = self.linear(x)#调用nn.Linear中的forwward
    return y_pred

#实例化/instance
model = LinearModel()

#损失函数
criterion = torch.nn.MSELoss()

#优化器  
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#optim优化模块，SGD类

#train
loss_list = []
epoch = 10
for i in range(epoch):
  y_pred = model(x_data)
  loss = criterion(y_pred,y_data)
  loss_list.append(loss)
  print(i,loss)
  optimizer.zero_grad()#梯度归零，要在backward之前
  loss.backward()#反向传播
  optimizer.step()#更新
  print('----------------------------------------------')
print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())


#可视化
plt.plot(range(epoch),loss_list)
plt.ylabel('epoch')
plt.xlabel('loss')
plt.show()
```
