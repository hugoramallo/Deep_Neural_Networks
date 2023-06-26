import torch
import torch.nn as nn
class Softmax(nn.Module):
    #out size, parámetro de tamaño de salida
    def __init__(self, in_size, out_size):
        super(Softmax, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
    def forward(self, x):
        out = self.linear(x)
        return out

from torch.nn import Linear

torch.manual_seed(1)

#el parámetro 2 se refiere a una mestra de entrada bidimensional
#primer parámetro imput features (2) y segundo parámetro clases (3)
model = Softmax(2,3)

#tensor con dos dimensiones
x = torch.tensor(([[1.0,2.0]]))

z = model(x)

yhat = z.max(1)


import torch
z = torch.tensor([[2, 5, 0], [10, 8, 2], [6, 5, 1]])
yhat = z.max(1)
print(yhat)




