
<h2 style='pointer-events: none;'>Getting Started with PyTorch</h2>
<h3 style='pointer-events: none;'>Common Operations on PyTorch Tensors</h3>

```
import torch
```
create an empty tensor
```
x = torch.empty(1, 1, 1, 1) # generate a four-dimensional tensor
y = torch.empty(1, 1) # generate a two-dimensional tensor
```
generate a random tensor
```
x = torch.rand(1, 1, 1, 1) # generate a four-dimensional tensor
y = torch.rand(1, 1) # generate a two-dimensional tensor
```
Element-Wise Operations in PyTorch
```
x = torch.rand(1, 1,) # generate a two-dimensional tensor
y = torch.rand(1, 1,) # generate a two-dimensional tensor
z_add = torch.add(x, y) # addition
z_sub = torch.sub(x, y) # subtraction
z_mul = torch.mul(x, y) # multiplication
z_div = torch.div(x, y) # division
```
Matrix Operations in PyTorch
```
x = torch.rand(1, 1, 1, 1) # generate a four-dimensional tensor
y = torch.rand(1, 1, 1, 1) # generate a four-dimensional tensor
z = torch.mm(x, y) # matrix multiplication
z = torch.t(x) # matrix transpose
```
PyTorch Indexing, Slicing, and Shape
```
x = torch.rand(5, 5, 5, 5) # generate a four-dimensional tensor
z = x[0, 0, 0, 0] # get the first element
z = x[0, 0, 0, :] # get all elements of the fourth dimension
z = x[0, 0, :, :] # get all elements of the third dimension
z = x[0, :, :, :] # get all elements of the second dimension
z = x[:, :, :, :] # get all elements of the first dimension
z = x[0:2, 0:2, 0:2, 0:2] # get the first two elements of each dimension
z = x.view(625) # convert a four-dimensional tensor to a one-dimensional tensor
z = x.view(25, 25) # convert a four-dimensional tensor to a two-dimensional tensor
x_shape = x.shape # get the shape of the tensor, same as size
x_shape = x.size() # get the shape of the tensor, same as shape
```
Concatenation of PyTorch Tensors
```
x = torch.rand(5, 5,) # generate a two-dimensional tensor
y = torch.rand(5, 5,) # generate a two-dimensional tensor
z = torch.cat((x, y), 0) # concatenate along the row
z = torch.cat((x, y), 1) # concatenate along the column
```
PyTorch Tensor Broadcasting Mechanism
```
x = torch.rand(5, 5,) # generate a two-dimensional tensor
y = torch.rand(5, 1,) # generate a two-dimensional tensor
z = x + y # broadcasting
```
Automatic Differentiation in PyTorch Tensors
```
x = torch.rand(2, 2, requires_grad=True) # requires_grad=True表示需要求导
y = torch.rand(2, 2, requires_grad=True)
z = x + y + 2
z = z.mean() # Calculate the mean, the gradient of the mean should be 1/n, where n is the number of elements
z.backward() # Backpropagation dz/dx and dz/dy
x_grad = x.grad # The gradient of x should be 1/4
y_grad = y.grad # The gradient of y should be 1/4
```
Concise Implementation of Linear Regression
```
import torch
# f = w * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# forward propagation
def forward(x):
    return w * x

# loss function is mse
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

# gradient descent
learning_rate = 0.01
n_iters = 100

# training loop
for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)
    # loss
    l = loss(Y, y_pred)
    # gradients = backward pass
    l.backward() # dl/dw
    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad
    # zero gradients
    w.grad.zero_()
    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
```
<h3 style='pointer-events: none;'>2.DataLoader and Dataset in pytorch</h3>

During training, we need to input data into the model in batches. This requires the use of DataLoader and Dataset. Dataset is an abstract class that needs to be inherited, and the **len** and **getitem** methods need to be implemented. The **len** method returns the size of the dataset, while the **getitem** method returns a sample from the dataset. DataLoader is an iterator that outputs the dataset in batches.

```
# training loop
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        ...
```
<h4 style='pointer-events: none;'>Dataset & DataLoader</h4>

Implement your own dataset by inheriting the Dataset class. Here we define a hydrological dataset specifically for the bp neural network.
```
import torch
import pandas as pd

class HydroDatasetForBP(torch.utils.data.Dataset):

    def __init__(self, data, target):
        data = pd.read_excel("Data/澴水流域数据.xlsx")
        # features start from the third column
        self.x = torch.tensor(data.iloc[:, 2:].values, dtype=torch.float32)
        # label is the second column
        self.y = torch.tensor(data.iloc[:, 1].values, dtype=torch.float32)
        # number of samples
        self.n_samples = data.shape[0]
        
    def __**getitem**__(self, index):
        return self.x[index], self.y[index]
    
    def __**len**__(self):
        return self.n_samples

dataset = HydroDatasetForBP()
dataloder = torch.utils.data.DataLoader(
                dataset=dataset, 
                batch_size=64, 
                shuffle=True,
                num_workers=2, # read data in multiple threads
            )
```
<h4 style='pointer-events: none;'>Training</h4>

```
num_epochs = 100
total_samples = **len**(dataset)
# number of iterations per epoch
n_iterations = math.ceil(total_samples/64)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward, update
        ...
        if (i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')
```
<h4 style='pointer-events: none;'>Dataset Transforms</h4>

In hydrology, the main focus is on processing runoff processes or various hydro-meteorological data, so it mainly involves tensor operations. You can also learn image processing as you wish. 
```
import torch
import pandas as pd
import torchvision
import torchtext

# copied from above
class HydroDatasetForBP(torch.utils.data.Dataset):

    def __init__(self, data, target, transform=None):
        data = pd.read_excel("Data/澴水流域数据.xlsx")
        # features start from the third column
        self.x = data.iloc[:, 2:].values
        # label is the second column
        self.y = data.iloc[:, 1].values
        # number of samples
        self.n_samples = data.shape[0]
        self.transform = transform
    
    def __**getitem**__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __**len**__(self):
        return self.n_samples
# write two transforms
class ToTensor:
    """
    transform data
    numpy array -> tensor
    """
    def __call__(self, data):
        inputs, targets = data
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class Scale:
    """
    scale data
    """
    def __init__(self, data_max)
        self.data_max = data_max
    
    def __call__(self, data):
        inputs, targets = data
        return inputs/self.data_max, targets/self.data_max

# combine two transforms
composed = torchvision.transforms.Compose([ToTensor(), Scale(1000)])
# create dataset
dataset = HydroDatasetForBP(transform=composed)
```
<h3 style='pointer-events: none;'>3.Pytorch Workflow</h3>

```
import torch
import torch.nn as nn
```
<h4 style='pointer-events: none;'>Problem definition</h4>

What should be the input and output of the model, how large should they be, and what type of problem is it 
```
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape # sample represents the number of samples, feature represents the number of features
input_size = n_features # input size
output_size = n_features # output size
```
<h4 style='pointer-events: none;'>Model design</h4>

According to the problem definition, what should the forward propagation process of the model be like?
```
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.lin(x)
model = LinearRegression(input_size, output_size)
```
Define the loss function and optimizer
```  
loss = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD([model.parameters()], lr=learning_rate)
```
<h4 style='pointer-events: none;'>Train the model</h4>

```
n_iters = 1000
for epoch in range(n_iters):
    y_pred = model(X)
    l = loss(Y, y_pred)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}') 
```
<h4 style='pointer-events: none;'>Test the model</h4>

```
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
```
<h4 style='pointer-events: none;'>Example: Linear Regression</h4>

```
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
```
Prepare the data
```
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
# change the dimension of Y from (100,) to (100,1)
Y = Y.view(Y.shape[0], 1)
n_samples, n_features = X.shape
```
Model design
```
input_size = n_features
output_size = 1
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.lin(x)
```
Loss function and optimizer
```
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```
Train the model
```
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, Y)
    # backward pass
    loss.backward()
    # update
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
```
Plot
```
# First convert X and Y to numpy arrays
predicted = model(X).detach().numpy()
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
```
<h4 style='pointer-events: none;'>Example: Logistic Regression</h4>

```
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
```
Prepare the data
```
# Use sklearn's dataset
bc = datasets.load_breast_cancer()
# Binary classification problem, 569 samples in total, 30 features per sample, y is 0 or 1
X, Y = bc.data, bc.target
n_samples, n_features = X.shape
# Divide the training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)
# Scale to [-1,1]
scaler = StandardScaler()
# Scale the features of the training set and test set to [-1,1]
X_train = scaler.fit_transform(X_train)
# Scale the features of the test set to [-1,1]
X_test = scaler.transform(X_test) # Pay attention here, this is not fit_transform because the scaling of the test set should use the scaler of the training set.
# Convert numpy arrays to tensors
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = Y_test.view(Y_test.shape[0], 1)
Y_train = Y_train.view(Y_train.shape[0], 1)
```
Model design
```
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        # 1 means the output dimension is 1
        self.lin = nn.Linear(n_input_features, 1)
    def forward(self, x):
        y_predicted = torch.sigmoid(self.lin(x))
        return y_predicted

model = LogisticRegression(n_features)
```
Loss function and optimizer
```
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```
Train the model
```
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, Y_train)
    # backward pass
    loss.backward()
    # update
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
```
Test the model
```
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(Y_test).sum() / float(Y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
```
