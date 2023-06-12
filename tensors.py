#1D tensors

import torch
import numpy as np
from torch._refs import item

a = torch.tensor([0,1,2,3,4,5])
#imprimir tipo de tensor, int64
print (a.dtype)
a.type()
torch.FloatTensor

#cambiar un tensor int a un float 32
a = a.type(torch.float32)
print(a.dtype)

#tamaño del tensor
print(a.size())

#dimensiones
print(a.ndimension()) #1 dimension es un vector


#remodelamos tensor usando view method
a_col=a.view(6,1)

#convertir numpy array a tensor
numpy_array = np.array([0.0,1.0,2.0,3.0,4.0])
torch_tensor = torch.from_numpy(numpy_array)
#convertir tensor a numpy
back_to_numpy = torch_tensor.numpy()
#imprimimos el tensor como una lista
print(a.tolist())
#cambiar un elemento en un tensor
a[0] = 100
print(a.tolist())
#sumar vectores
u = torch.tensor([1.0,0.0])
v = torch.tensor([0.0,1.0])
z = u + v
#resultado
print (z.tolist())

#resultado producto hadamard (multiplicación)
u = torch.tensor([1,2])
v = torch.tensor([3,1])
z = u * v
result = torch.dot(u,v)
print(result)
#radiofusion
u = torch.tensor([1,2,3-1])
z = u + 1
print(u.tolist())

#ver el valor máximo
b = torch.tensor([1,-2,3,4,5])
max_b=b.max()
print(max_b)
#linespace devuelve los números espaciados
#podemos usarla para generar muestras uniformemente espaciadas

torch.linspace(-2,2,steps=5)

x = torch.linspace(0,2*np.pi,100)
#calculamos seno de X
y = torch.sin(x)
#matplotlib para imprimir
import matplotlib.pyplot as plt
#%matplotlib inline
#imprimimos la función
plt.plot(x.numpy(),y.numpy())

#obtiene el valor de un tensor de un solo elemento
value = torch_tensor.item()





