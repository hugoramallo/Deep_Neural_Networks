
import torch

my_tensor = torch.tensor([[1,2,3], [8,7,6], [3,2,1]]) # 3(filas) x 3(columnas) tensor matrix
#tamaño del tensor
print(my_tensor.size())
#calcula el número de elementos
print(my_tensor.numel()) #elementos 9
#imprimir el segundo valor en la segunda fila
print(my_tensor[1][1])
#extraer dos elementos de una matriz de tensores usando slicing
print(my_tensor[1:3,2])
#mostrar los tensores en formato matriz
for row in my_tensor:
 print(row.tolist())
print('\n')

my_tensor2 = torch.tensor([[5,4], [8,7], [3,2]]) # 3(filas) x 2(columnas) tensor matrix
for row in my_tensor2:
    print(row.tolist())

#multiplicar matrices de dos tensores
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
#multiplicación
C = torch.mm(A, B)
#resultado
print(C)