
<h2 style='pointer-events: none;'>Basic NN for classification</h2>
<h3 style='pointer-events: none;'>1.Forward Neural network</h3>

Simple feedforward neural network for MNIST dataset classification using PyTorch.
<h4 style='pointer-events: none;'>Dataset</h4>

MINIST dataset is used for this example. It contains 60,000 training images and 10,000 testing images of handwritten digits. The images are grayscale and 28x28 pixels in size. The dataset is divided into 10 classes, one for each digit.
```
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/MNIST',
                                           train=True,
                                           download=True,
                                           transform=transforms.ToTensor()
                                           )
test_dataset = torchvision.datasets.MNIST(root='./data/MNIST',
                                            train=False,
                                            transform=transforms.ToTensor()
                                            )
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True
                                           )
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False
                                            )
```
<h4 style='pointer-events: none;'>Model</h4>

neural network, each hidden layer contains 256 neurons, activation function is ReLU, output layer contains 10 neurons, activation function is Softmax.
```
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        # out = self.softmax(out) # CrossEntropyLoss() has included softmax
        return out
        
model = NeuralNet(input_size, hidden_size, num_classes)
```
<h4 style='pointer-events: none;'>Loss function and optimizer</h4>

CrossEntropyLoss() is used for the loss function, and Adam optimizer is used for the optimizer.
```
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```
<h4 style='pointer-events: none;'>Training</h4>

```
total_step = len(train_loader)
for epoch in range (num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, 28*28) # -1 means the size is inferred
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
```
<h4 style='pointer-events: none;'>Testing</h4>

```
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
```
<h3 style='pointer-events: none;'>2.Cross-entropy and softmax regression</h3>
<h4 style='pointer-events: none;'>Cross-entropy</h4>

Cross-entropy is generally suitable for measuring the difference between two probability distributions. It is a concept in information theory used to quantify the distance between two probability distributions. The smaller the cross-entropy, the closer the two probability distributions. The definition of cross-entropy is as follows:
$$
H(p,q)=-\sum_{x}p(x)logq(x)
$$
```
import numpy as np

def cross_entropy(actual, predicted):
    return - np.sum(actual * np.log(predicted))
```
Cross-entropy loss calculates the distance between two probability distributions, not the distance between the true labels and predicted values. Therefore, cross-entropy loss is more suitable for classification tasks than mean squared error loss. An example using cross-entropy loss is:
```
# One hot encoding
Y = np.array([1, 0, 0])
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
```
<h4 style='pointer-events: none;'>softmax regression</h4>

Softmax regression is a multi-class model, and its output is a probability distribution. Assuming there are K categories, for an input sample x, the probability of it belonging to the k-th class is:
$$
P(y=k|x)=\frac{exp(x^Tw_k)}{\sum_{i=1}^Kexp(x^Tw_i)}
$$
<h4 style='pointer-events: none;'>Implementation with Pytorch</h4>

```
import torch
import torch.nn as nn
import torch.nn.functional as F

loss = nn.CrossEntropyLoss()
Y = torch.tensor([0])
# nsamples x nclasses = 1 x 4
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1, 0.5]])
Y_pred_bad = torch.tensor([[0.5, 1.0, 2.1, 0.3]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())

# get predicted classes
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(predictions1)
print(predictions2)
```
<h4 style='pointer-events: none;'>A simple multi-classification task</h4>

```
import torch
import torch.nn as nn

# Multiclass problem
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no activation and no softmax at the end
        return out
        # if it's a binary classification problem, we can use sigmoid
        # y_pred = torch.sigmoid(out)
        # return y_pred

model = NeuralNet(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss() # applies softmax for us
# criterion = nn.BCELoss() # for binary classification
```
<h3 style='pointer-events: none;'>3.Concise implementation of softmax regression</h3>

The softmax function transforms outputs into a valid probability distribution, making softmax regression suitable for classification problems. The softmax function turns the unnormalized predictions into non-negative values that sum to 1, while maintaining differentiability for the model. To achieve this, we first exponentiate each unnormalized prediction to ensure non-negativity. To ensure the final output probabilities sum to 1, we then divide each exponentiated result by their sum:
$$
\hat{\mathbf{y}}=\operatorname{softmax}(\mathbf{o}) \quad \text  \quad \hat{y}_j=\frac{\exp \left(o_j\right)}{\sum_k \exp \left(o_k\right)}
$$
Here, $\hat{y}_j$ represents the probability of predicting $y_j$, and $\mathbf{o}$ is the unnormalized prediction we aim to obtain. Therefore, the softmax function here essentially serves as a normalization:
<h4 style='pointer-events: none;'>Image classification dataset (Fashion-MNIST)</h4>

```python
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()
```
Fashion-MNIST dataset consists of images where each input image has a height and width of 28 pixels. The dataset consists of grayscale images with a channel size of 1.


```python
# Use the ToTensor instance to transform the image data from the PIL type to the 32-bit floating point format and divide by 255 so that the numerical value of all pixels
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
mnist_train[0][0].shape
# The result is torch.Size([1, 28, 28]), indicating that the number of channels in the image in the dataset is 1, and the height and width are both 28 pixels
```

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ../data\FashionMNIST\raw\train-images-idx3-ubyte.gz
    

    100%|██████████| 26421880/26421880 [00:04<00:00, 5308358.83it/s] 
    

    Extracting ../data\FashionMNIST\raw\train-images-idx3-ubyte.gz to ../data\FashionMNIST\raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ../data\FashionMNIST\raw\train-labels-idx1-ubyte.gz
    

    100%|██████████| 29515/29515 [00:00<00:00, 42763.88it/s]
    

    Extracting ../data\FashionMNIST\raw\train-labels-idx1-ubyte.gz to ../data\FashionMNIST\raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ../data\FashionMNIST\raw\t10k-images-idx3-ubyte.gz
    

    100%|██████████| 4422102/4422102 [00:25<00:00, 172989.88it/s] 
    

    Extracting ../data\FashionMNIST\raw\t10k-images-idx3-ubyte.gz to ../data\FashionMNIST\raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ../data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz
    

    100%|██████████| 5148/5148 [00:00<00:00, 1416909.05it/s]

    Extracting ../data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz to ../data\FashionMNIST\raw
    

    
    




    torch.Size([1, 28, 28])



label is a number, which represents the clothing category represented by the image. We can use the following function to obtain the name of the clothing category represented by the number label.


```python
def get_fashion_mnist_labels(labels):  #@save
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = [
        "t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt",
        "sneaker", "bag", "ankle boot"
    ]
    return [text_labels[int(i)] for i in labels]
```
Here we can see the size of the training set and the test set.


```python
len(mnist_train), len(mnist_test)
```




    (60000, 10000)



Define a function to display images, which is convenient for us to view the images in the dataset.

```python
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

# 取一个批量的样本，一共18张图像
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
```
<h4 style='pointer-events: none;'>softmax regression model</h4>

```python
from torch import nn
batch_size = 256
train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
```
Initialize the model

```python
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
# Initialize the weight
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```
Define the loss function and the optimization algorithm

```python
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```
Train the model

```python
num_epochs = 10
for epoch in range(num_epochs):
    for X, y in train_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.mean().backward() # mean loss of a batch of samples
        trainer.step()
    l = loss(net(X), y)
    print(f'epoch {epoch + 1}, loss {l:f}')
```
<h3 style='pointer-events: none;'>4.Use a Convolutional Neural Network (CNN) for the classification task</h3>
<h4 style='pointer-events: none;'>CNN</h4>

First, let's understand the concept of convolution. The convolution formula looks like this  
$$  
y(t)=\int_{-\infty}^{\infty} x(\tau) h(t-\tau) d \tau
$$
The detailed meaning can be found in [this video](https://www.youtube.com/watch?v=D641Ucd_xuw). 
It's worth noting that this formula can be understood as follows: If a system has an unstable input and produces a stable output, we can use convolution to find the system's inventory.
For example, the calculation of surface runoff using the unit line of the time period is actually a convolution operation.  
The net rainfall at each time is unstable, but no matter how much net rainfall was input, the operation on these rainfalls is fixed, that is, the net rainfall at each time is multiplied by a unit line, and then summed. The unit line is time-invariant and is a fixed function, so we can naturally calculate the surface runoff. Naturally, the surface runoff at each time is actually the inventory of the surface runoff at that time, and the surface runoff at that time is determined by the net rainfall at all previous times.  
Here, the unit line is the convolution kernel, and the convolution kernel is a fixed function that operates on the input data.
Now we understand the convolution in a narrow sense, but the convolution operation of CNN is actually more general. What we just understood is the operation that occurs in the time domain, but what if $t$ here does not refer to the time domain but to the distance(time distance, space distance)? ,
whatever, and the input function here may not be one-dimensional, the convolution operation can be generalized to:
$$
y(x,y)=\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} x(\tau_1,\tau_2) h(x-\tau_1,y-\tau_2) d \tau_1 d \tau_2
$$
x,y are actually the coordinates of the pixel point in the image corresponding to the center of the convolution kernel,

|         | 1st column | 2nd column | 3rd column |
|---------|--------------|-------------|--------------|
| **1st row** | $f(x-1,y-1)$ | $f(x-1,y)$  | $f(x-1,y+1)$ |
| **2nd row** | $f(x,y-1)$   | $f(x,y)$    | $f(x,y+1)$   |
| **3rd row** | $f(x+1,y-1)$ | $f(x+1,y)$  | $f(x+1,y+1)$ |

Because the pixel points are not continuous, the integral here can be understood as a sum, so we can get a convolution kernel, the function of this convolution kernel is to perform a fixed operation on the input data.
<h4 style='pointer-events: none;'>CNN for images classification</h4>

CNN is primarily used for capturing local features in images. While I may not fully understand why they work for time-series modeling like rainfall-runoff simulation, they do prove effective sometimes. This suggests that the convolution operation here may have certain properties that enable it to capture temporal relationships. For now, let's implement CNN for image classification, as this type of work might be useful in future applications such as rainfall-runoff simulation using remote sensing images.  
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
```
First, if we have a GPU, we use the GPU, because this is an image classification problem, and the GPU will be much faster.
```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
<h4 style='pointer-events: none;'>Set some hyperparameters  </h4>

```
num_epochs = 200
batch_size = 4
learning_rate = 0.001
```
<h4 style='pointer-events: none;'>Dataset and Dataloader</h4>

```
# the transform contains two operations: ToTensor() converts the image to Tensor, Normalize() does normalization
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True)
                                                       
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                shuffle=False)
                               
classes = ('plane', 'car', 'bird', 'cat',
             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```
<h4 style='pointer-events: none;'>Model</h4>

```
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 chanels, 6 kernels, kernel size 5
        self.pool = nn.MaxPool2d(2, 2) # kernel size 2, stride 2
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 chanels, 16 kernels, kernel size 5
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 16*5*5 is the size of the output of the second convolution layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes) # num_classes is 10
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 32*32*3 -> 28*28*6 -> 14*14*6
        x = self.pool(F.relu(self.conv2(x))) # 14*14*6 -> 10*10*16 -> 5*5*16
        x = x.view(-1, 16 * 5 * 5) # 5*5*16 -> 400
        x = F.relu(self.fc1(x)) # 400 -> 120
        x = F.relu(self.fc2(x)) # 120 -> 84
        x = self.fc3(x) # 84 -> 10
        return x
model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```
<h4 style='pointer-events: none;'>Training</h4>

```
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images
        labels = labels
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 1000 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
```
<h4 style='pointer-events: none;'>Testing</h4>

```
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images
        labels = labels
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
```

