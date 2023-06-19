# Import the libraries we need for this lab

import torch.nn as nn
import torch
import matplotlib.pyplot as plt

#random seed
# Set the random seed

torch.manual_seed(2)

#tensor ranging
z = torch.arange(-100, 100, 0.1).view(-1, 1)
print("The tensor: ", z)

#sigmoid object
# Create sigmoid object

sig = nn.Sigmoid()

#element-wise function sigmoid
# Use sigmoid object to calculate the

yhat = sig(z)

#plot
plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('z')
plt.ylabel('yhat')

#Apply the element-wise Sigmoid from the function module and plot the results
yhat = torch.sigmoid(z)
plt.plot(z.numpy(), yhat.numpy())

#logistic regression with nn.sequential

# Create x and X tensor

x = torch.tensor([[1.0]])
X = torch.tensor([[1.0], [100]])
print('x = ', x)
print('X = ', X)

#logistic regression object
# Use sequential function to create model

model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

#print parameters
# Print the parameters

print("list(model.parameters()):\n ", list(model.parameters()))
print("\nmodel.state_dict():\n ", model.state_dict())

#prediction for x
# The prediction for x

yhat = model(x)
print("The prediction: ", yhat)

#predictions multiple samples
# The prediction for X

yhat = model(X)
yhat

# Create and print samples

x = torch.tensor([[1.0, 1.0]])
X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
print('x = ', x)
print('X = ', X)

# Create new model using nn.sequential()

model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

# Print the parameters

print("list(model.parameters()):\n ", list(model.parameters()))
print("\nmodel.state_dict():\n ", model.state_dict())

# Make the prediction of x

yhat = model(x)
print("The prediction: ", yhat)

# The prediction of X

yhat = model(X)
print("The prediction: ", yhat)

#custom modules

# Create logistic_regression custom class

class logistic_regression(nn.Module):

    # Constructor
    def __init__(self, n_inputs):
        super(logistic_regression, self).__init__()
        self.linear = nn.Linear(n_inputs, 1)

    # Prediction
    def forward(self, x):
        yhat = torch.sigmoid(self.linear(x))
        return yhat

# Create x and X tensor

x = torch.tensor([[1.0]])
X = torch.tensor([[-100], [0], [100.0]])
print('x = ', x)
print('X = ', X)

# Create logistic regression model

model = logistic_regression(1)

# Print parameters

print("list(model.parameters()):\n ", list(model.parameters()))
print("\nmodel.state_dict():\n ", model.state_dict())

# Make the prediction of x

yhat = model(x)
print("The prediction result: \n", yhat)


# Make the prediction of X

yhat = model(X)
print("The prediction result: \n", yhat)

# Create logistic regression model

model = logistic_regression(2)

# Create x and X tensor

x = torch.tensor([[1.0, 2.0]])
X = torch.tensor([[100, -100], [0.0, 0.0], [-100, 100]])
print('x = ', x)
print('X = ', X)

# Make the prediction of x

yhat = model(x)
print("The prediction result: \n", yhat)

# Make the prediction of X

yhat = model(X)
print("The prediction result: \n", yhat)

# Practice: Make your model and make the prediction

X = torch.tensor([-10.0])
my_model = nn.Sequential(nn.Linear(1, 1),nn.Sigmoid())
yhat = my_model(X)
print("The prediction: ", yhat)
