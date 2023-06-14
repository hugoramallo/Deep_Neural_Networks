# These are the libraries will be used for this lab.

import numpy as np
import matplotlib.pyplot as plt
import torch

"""
The class helps us to visualize the data space and 
the parameter space during training and has nothing to do with PyTorch."""


# The class for plotting

class plot_diagram():

    # Constructor
    def __init__(self, X, Y, w, stop, go=False):
        start = w.data
        self.error = []
        self.parameter = []
        print(type(X.numpy()))
        self.X = X.numpy()

        self.Y = Y.numpy()
        self.parameter_values = torch.arange(start, stop)
        self.Loss_function = [criterion(forward(X), Y) for w.data in self.parameter_values]
        w.data = start

    # Executor
    def __call__(self, Yhat, w, error, n):
        self.error.append(error)
        self.parameter.append(w.data)
        plt.subplot(212)
        plt.plot(self.X, Yhat.detach().numpy())
        plt.plot(self.X, self.Y, 'ro')
        plt.xlabel("A")
        plt.ylim(-20, 20)
        plt.subplot(211)
        plt.title("Data Space (top) Estimated Line (bottom) Iteration " + str(n))
        # Convert lists to PyTorch tensors
        parameter_values_tensor = torch.tensor(self.parameter_values)
        loss_function_tensor = torch.tensor(self.Loss_function)

        # Plot using the tensors
        plt.plot(parameter_values_tensor.numpy(), loss_function_tensor.numpy())

        plt.plot(self.parameter, self.error, 'ro')
        plt.xlabel("B")
        plt.figure()

    # Destructor
    def __del__(self):
        plt.close('all')

    # Import the library PyTorch
    import torch
    #Generate values from -3 to 3 that create a line with a slope of -3. This is the line you will estimate.

    # Create the f(X) with a slope of -3
    X = torch.arange(-3, 3, 0.1).view(-1, 1)
    f = -3 * X

    # Plot the line with blue
    plt.plot(X.numpy(), f.numpy(), label='f')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


    """ Let us add some noise to the data in order to simulate the real data. Use torch.randn(X.size()) 
    to generate Gaussian noise that is the same size as X and has a standard deviation opf 0.1."""

    # Add some noise to f(X) and save it in Y
    Y = f + 0.1 * torch.randn(X.size())

    # Plot the data points (Y)

    plt.plot(X.numpy(), Y.numpy(), 'rx', label='Y')

    plt.plot(X.numpy(), f.numpy(), label='f')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    #Create the Model and Cost Function (Total Loss)

    #First, define the <code>forward</code> function $y=w*x$. (We will add the bias in the next lab.)

    # Create forward function for prediction
    def forward(x):
        return w * x

    # Create the MSE (MEAN SQUARE ERROR) function for evaluate the result.
    def criterion(yhat, y):
        return torch.mean((yhat - y) ** 2)

    #Define the learning rate lr and an empty list LOSS to record the loss for each iteration:

    #Create Learning Rate and an empty list to record the loss for each iteration
    lr = 0.1
    LOSS = []

    #Now, we create a model parameter by setting the argument requires_grad to  True because the system must learn it.
    w = torch.tensor(-10.0, requires_grad=True)

    #Create a <code>plot_diagram</code> object to visualize the data space and the parameter space for each iteration during training:
    gradient_plot = plot_diagram(X, Y, w, stop=5)

    # Define a function for train the model

    def train_model(iter):
        for epoch in range(iter):
            # make the prediction as we learned in the last lab
            Yhat = forward(X)

            # calculate the iteration
            loss = criterion(Yhat, Y)

            # plot the diagram for us to have a better idea
            gradient_plot(Yhat, w, loss.item(), epoch)

            # store the loss into list
            LOSS.append(loss.item())

            # backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward()

            # updata parameters
            w.data = w.data - lr * w.grad.data

            # zero the gradients before running the backward pass
            w.grad.data.zero_()

    # Give 4 iterations for training the model here.

    train_model(4)

    # Plot the loss for each iteration

    plt.plot(LOSS)
    plt.tight_layout()
    plt.xlabel("Epoch/Iterations")
    plt.ylabel("Cost")

    # Practice: Create w with the inital value of -15.0
    # Type your code here
    w = torch.tensor(-15.0, requires_grad=True)

    # Practice: Create LOSS2 list

    # Type your code here
    LOSS2 = []

    # Practice: Create your own my_train_model
    gradient_plot1 = plot_diagram(X, Y, w, stop=15)


    def my_train_model(iter):
        for epoch in range(iter):
            # make the prediction as we learned in the last lab
            Yhat = forward(X)
            # calculate the iteration
            loss = criterion(Yhat, Y)
            # plot the diagram for us to have a better idea
            gradient_plot(Yhat, w, loss.item(), epoch)
            # store the loss into list
            LOSS.append(loss.item())
            # backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward()
            # updata parameters
            w.data = w.data - lr * w.grad.data
            # zero the gradients before running the backward pass
            w.grad.data.zero_()

    # Give 4 iterations for training the model here.

    train_model(4)

    plt.plot(LOSS, label="LOSS")
    plt.plot(LOSS2, label="LOSS2")
    plt.tight_layout()
    plt.xlabel("Epoch/Iterations")
    plt.ylabel("Cost")
    plt.legend()