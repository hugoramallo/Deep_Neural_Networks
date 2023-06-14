# These are the libraries will be used for this lab.

import torch

"""Let us create the following expressions:

𝑏=−1,𝑤=2
 
𝑦̂ =−1+2𝑥"""

# Define w = 2 and b = -1 for y = wx + b

w = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(-1.0, requires_grad = True)

#Then, define the function forward(x, w, b) makes the prediction:
# Function forward(x) for prediction
#linear regression formula
def forward(x):
    yhat = w * x + b
    return yhat

"""Let's make the following prediction at x = 1

𝑦̂ =−1+2𝑥
 
𝑦̂ =−1+2(1)"""

# Predict y = 2x - 1 at x = 1

x = torch.tensor([[1.0]])
yhat = forward(x)
print("The prediction: ", yhat)

#Let us construct the x tensor first. Check the shape of x.

# Create x Tensor and check the shape of x tensor
x = torch.tensor([[1.0], [2.0]])
print("The shape of x: ", x.shape)

#Now make the prediction

# Make the prediction of y = 2x - 1 at x = [1, 2]
yhat = forward(x)
print("The prediction: ", yhat)

#Make a prediction of the following x tensor using the w and b from above.
# Practice: Make a prediction of y = 2x - 1 at x = [[1.0], [2.0], [3.0]]
x = torch.tensor([[1.0], [2.0], [3.0]])

# Import Class Linear
from torch.nn import Linear

#Set the random seed because the parameters are randomly initialized
# Set random seed
torch.manual_seed(1)


# Create Linear Regression Model, and print out the parameters
lr = Linear(in_features=1, out_features=1, bias=True)
print("Parameters w and b: ", list(lr.parameters()))

"""
This is equivalent to the following expression:
𝑏=−0.44,𝑤=0.5153
 
𝑦̂ =−0.44+0.5153𝑥"""

#A method state_dict() Returns a Python dictionary object corresponding to the layers of each parameter tensor.
print("Python dictionary: ",lr.state_dict())
print("keys: ",lr.state_dict().keys())
print("values: ",lr.state_dict().values())

#The keys correspond to the name of the attributes and the values correspond to the parameter value.
print("weight:",lr.weight)
print("bias:",lr.bias)

#Now let us make a single prediction at x = [[1.0]].
# Make the prediction at x = [[1.0]]

x = torch.tensor([[1.0]])
yhat = lr(x)
print("The prediction: ", yhat)

#Use model lr(x) to predict the result.

# Create the prediction using linear model
x = torch.tensor([[1.0], [2.0]])
yhat = lr(x)
print("The prediction: ", yhat)

#Make a prediction of the following x tensor using the linear regression model lr.

# Practice: Use the linear regression model object lr to make the prediction.
x = torch.tensor([[1.0],[2.0],[3.0]])

# Practice: Make a prediction of y = 2x - 1 at x = [[1.0], [2.0], [3.0]]
x = torch.tensor([[1.0], [2.0], [3.0]])
yhat = forward(x)
print('The prediction: ', yhat)

#Set the random seed because the parameters are randomly initialized:
# Set random seed, random numbers
torch.manual_seed(1)

"""Let us create the linear object by using the constructor. The parameters are randomly created. 
Let us print out to see what w and b. The parameters of an torch.nn.Module model are contained in the model’s parameters accessed with lr.parameters():"""

# Create Linear Regression Model, and print out the parameters
lr = Linear(in_features=1, out_features=1, bias=True)
print("Parameters w and b: ", list(lr.parameters()))

"""This is equivalent to the following expression:

𝑏=−0.44,𝑤=0.5153
 
𝑦̂ =−0.44+0.5153𝑥"""

#A method state_dict() Returns a Python dictionary object corresponding to the layers of each parameter tensor.
print("Python dictionary: ",lr.state_dict())
print("keys: ",lr.state_dict().keys())
print("values: ",lr.state_dict().values())

#The keys correspond to the name of the attributes and the values correspond to the parameter value.
print("weight:",lr.weight)
print("bias:",lr.bias)

#Now let us make a single prediction at x = [[1.0]].

# Make the prediction at x = [[1.0]]
x = torch.tensor([[1.0]])
yhat = lr(x)
print("The prediction: ", yhat)

#Use model lr(x) to predict the result.

# Create the prediction using linear model
x = torch.tensor([[1.0], [2.0]])
yhat = lr(x)
print("The prediction: ", yhat)

#Make a prediction of the following x tensor using the linear regression model lr.

# Practice: Use the linear regression model object lr to make the prediction.
x = torch.tensor([[1.0],[2.0],[3.0]])
yhat = lr(x)
print("The prediction: ", yhat)

#Build Custom Modules

# Library for this section
from torch import nn


# Customize Linear Regression Class
class LR(nn.Module):

    # Constructor
    def __init__(self, input_size, output_size):
        # Inherit from parent
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    # Prediction function
    def forward(self, x):
        out = self.linear(x)
        return out

    # Create the linear regression model. Print out the parameters.
    lr = LR(1, 1)
    print("The parameters: ", list(lr.parameters()))
    print("Linear model: ", lr.linear)

    # Try our customize linear regression model with single input
    x = torch.tensor([[1.0]])
    yhat = lr(x)
    print("The prediction: ", yhat)

    # Try our customize linear regression model with multiple input
    x = torch.tensor([[1.0], [2.0]])
    yhat = lr(x)
    print("The prediction: ", yhat)

    #the parameters are also stored in an ordered dictionary :
    print("Python dictionary: ", lr.state_dict())
    print("keys: ", lr.state_dict().keys())
    print("values: ", lr.state_dict().values())

    # Practice: Use the LR class to create a model and make a prediction of the following tensor.
    x = torch.tensor([[1.0], [2.0], [3.0]])
    lr1 = LR(1, 1)
    yhat = lr(x)
    yhat