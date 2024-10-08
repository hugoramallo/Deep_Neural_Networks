# These are the libraries will be used for this lab.

import torch
from torch.utils.data import Dataset
torch.manual_seed(1)


# Define class for dataset

class toy_set(Dataset):

    # Constructor with defult values
    def __init__(self, length=100, transform=None):
        self.len = length
        self.x = 2 * torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        self.transform = transform

    # Getter
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    # Get Length
    def __len__(self):
        return self.len

# Create Dataset Object. Find out the value on index 1. Find out the length of Dataset Object.

our_dataset = toy_set()
print("Our toy_set object: ", our_dataset)
print("Value on index 0 of our toy_set object: ", our_dataset[0])
print("Our toy_set length: ", len(our_dataset))

# Use loop to print out first 3 elements in dataset

for i in range(3):
    x, y=our_dataset[i]
    print("index: ", i, '; x:', x, '; y:', y)

for x,y in our_dataset:
    print(' x:', x, 'y:', y)

my_dataset = toy_set(length = 50)
print("My toy_set length: ", len(my_dataset))


# Create tranform class add_mult
class add_mult(object):

    # Constructor
    def __init__(self, addx=1, muly=2):
        self.addx = addx
        self.muly = muly

    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x + self.addx
        y = y * self.muly
        sample = x, y
        return sample

# Create an add_mult transform object, and an toy_set object

a_m = add_mult()
data_set = toy_set()

# Use loop to print out first 10 elements in dataset

for i in range(10):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = a_m(data_set[i])
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)

# Create a new data_set object with add_mult object as transform

cust_data_set = toy_set(transform = a_m)

# Use loop to print out first 10 elements in dataset

for i in range(10):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = cust_data_set[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)


class my_add_mult(object):
    def __init__(self, add=2, mul=10):
        self.add = add
        self.mul = mul

    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x + self.add
        y = y + self.add
        x = x * self.mul
        y = y * self.mul
        sample = x, y
        return sample


my_dataset = toy_set(transform=my_add_mult())
for i in range(3):
    x_, y_ = my_dataset[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)

# Run the command below when you do not have torchvision installed
# !mamba install -y torchvision

from torchvision import transforms


# Create tranform class mult

class mult(object):

    # Constructor
    def __init__(self, mult=100):
        self.mult = mult

    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x * self.mult
        y = y * self.mult
        sample = x, y
        return sample

# Combine the add_mult() and mult()

data_transform = transforms.Compose([add_mult(), mult()])
print("The combination of transforms (Compose): ", data_transform)

data_transform(data_set[0])

x,y=data_set[0]
x_,y_=data_transform(data_set[0])
print( 'Original x: ', x, 'Original y: ', y)

print( 'Transformed x_:', x_, 'Transformed y_:', y_)

# Create a new toy_set object with compose object as transform

compose_data_set = toy_set(transform = data_transform)

# Use loop to print out first 3 elements in dataset

for i in range(3):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = cust_data_set[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
    x_co, y_co = compose_data_set[i]
    print('Index: ', i, 'Compose Transformed x_co: ', x_co ,'Compose Transformed y_co: ',y_co)

my_compose = transforms.Compose([mult(), add_mult()])
my_transformed_dataset = toy_set(transform = my_compose)
for i in range(3):
    x_, y_ = my_transformed_dataset[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)