<h2 style='pointer-events: none;'>Basic NN for Runoff Prediction</h2>
<h3 style='pointer-events: none;'>1.Using a bp neural network for runoff prediction</h3>

```
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
```
<h4 style='pointer-events: none;'>Load the dataset</h4>

```
Data = pd.read_excel("Data/data_bp.xlsx")
Data = Data.values
Data = torch.from_numpy(Data)
# use the last data of each row to predict the last data of next row
x = Data[0:-1, :]
y = Data[1:, -1]
# dataset and dataloader
batch_size = 100
train_dataset = torch.utils.data.TensorDataset(x, y)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2,
                                           )
```
<h4 style='pointer-events: none;'>Define the network</h4>

```
class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

input_size = 3
output_size = 1
model = NeuralNet(input_size, output_size)
```
<h4 style='pointer-events: none;'>Define the loss function and optimizer</h4>

```
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
```
<h4 style='pointer-events: none;'>Train the network</h4>

```
num_epochs = 200
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(train_loader):
        # Forward propagation
        x = x.reshape(-1, 3)
        y = y.reshape(-1, 1)
        outputs = model(x.float())
        loss = criterion(outputs, y.float())
        loss_list.append(loss.item())
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print the loss every 100 steps
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
```
<h4 style='pointer-events: none;'>Plot the loss</h4>

```
plt.plot(loss_list)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss function')
plt.show()

```
<h3 style='pointer-events: none;'>2.Implement runoff prediction using LSTM</h3>
<h4 style='pointer-events: none;'>Import lib</h4>

```
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
```
<h4 style='pointer-events: none;'>Read data</h4>

```
data = pd.read_excel("Data/LSTM神经网络数据.xlsx", index_col=0)
target_column = "径流量"
features = data.drop(target_column, axis=1)
target = data[target_column]
```
<h4 style='pointer-events: none;'>Normalize the data</h4>

```
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
features = x_scaler.fit_transform(features)
target = y_scaler.fit_transform(target.values.reshape(-1, 1))
```
<h4 style='pointer-events: none;'>Prepare input and output for LSTM</h4>

```
class TimeSeriesDataset(Dataset):
def __init__(self, sequences, targets):
self.sequences = sequences
self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return (
            self.sequences[index],
            self.targets[index]
        )

def create_sequences(features, target, time_steps):
xs = []
ys = []

    for i in range(len(features) - time_steps):
        xs.append(features[i : i + time_steps])
        ys.append(target[i + time_steps])
    # convert data to numpy arrays
    xs = np.array(xs)
    ys = np.array(ys)
    # convert data to tensors
    xs = torch.from_numpy(xs).float()
    ys = torch.from_numpy(ys).float()
    return xs, ys

time_steps = 30
x, y = create_sequences(features, target, time_steps)
```
<h4 style='pointer-events: none;'>Split the dataset into training and testing sets</h4>

```
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1
x_train = x[:int(len(x) * train_ratio)]
y_train = y[:int(len(y) * train_ratio)]
x_valid = x[int(len(x) * train_ratio):int(len(x) * (train_ratio + valid_ratio))]
y_valid = y[int(len(y) * train_ratio):int(len(y) * (train_ratio + valid_ratio))]
x_test = x[int(len(x) * (train_ratio + valid_ratio)):]
y_test = y[int(len(y) * (train_ratio + valid_ratio)):]
```
<h4 style='pointer-events: none;'>Create a Dataset and DataLoader</h4>

```
train_data = TimeSeriesDataset(x_train, y_train)
valid_data = TimeSeriesDataset(x_valid, y_valid)
test_data = TimeSeriesDataset(x_test, y_test)

batch_size = 128
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size)
```
<h4 style='pointer-events: none;'>Create the LSTM model</h4>

```
class LSTM(nn.Module):
def __init__(self, input_size, lstm_hidden_size, lstm_num_layers, output_size, num_fc_layers, fc_size):
super(LSTM, self).__init__()
self.hidden_size = lstm_hidden_size
self.num_layers = lstm_num_layers

        # LSTM
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, lstm_num_layers, batch_first=True)

        # fully connected layers
        fc_layers = [nn.Linear(lstm_hidden_size if i == 0 else fc_size, fc_size) for i in range(num_fc_layers)]
        self.fc_layers = nn.ModuleList(fc_layers)
        self.final_fc = nn.Linear(fc_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]

        # fully connected layers
        for fc in self.fc_layers:
            out = torch.relu(fc(out))

        out = self.final_fc(out)
        return out

model = LSTM(
input_size=x_train.shape[-1],
lstm_hidden_size=64,
lstm_num_layers=3,
output_size=1,
num_fc_layers=3,
fc_size=32
)
```
<h4 style='pointer-events: none;'>Loss function and optimizer</h4>

```
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
```
<h4 style='pointer-events: none;'>Train the model</h4>

```
epochs = 300
train_losses = []
valid_losses = []
for epoch in range(epochs):
batch_losses = []
for x_batch, y_batch in train_loader:
optimizer.zero_grad()
output = model(x_batch)
loss = criterion(output, y_batch)
loss.backward()
optimizer.step()
batch_losses.append(loss.item())
train_losses.append(np.mean(batch_losses))
print(f"Epoch {epoch} train loss: {train_losses[-1]}")

    with torch.no_grad():
        batch_losses = []
        for x_batch, y_batch in valid_loader:
            output = model(x_batch)
            loss = criterion(output, y_batch)
            batch_losses.append(loss.item())
        valid_losses.append(np.mean(batch_losses))
        # save the best model
        if epoch == 0 or valid_losses[-1] < min(valid_losses[:-1]):
            best_model = model.state_dict().copy()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} train loss: {train_losses[-1]} valid loss: {valid_losses[-1]}")
```
<h4 style='pointer-events: none;'>Plot loss</h4>

```
plt.plot(train_losses, label="Training loss")
plt.plot(valid_losses, label="Validation loss")
plt.legend()
plt.show()
```
<h4 style='pointer-events: none;'>Test the model</h4>

```
model.load_state_dict(best_model)
with torch.no_grad():
test_losses = []
for x_batch, y_batch in valid_loader:
output = model(x_batch)
loss = criterion(output, y_batch)
output = y_scaler.inverse_transform(output.numpy())
y_batch = y_scaler.inverse_transform(y_batch.numpy())
test_losses.append(loss.item())
print(f"Test loss: {np.mean(test_losses)}")
```
<h3 style='pointer-events: none;'>3.Using Transformer for runoff prediction</h3>
<h4 style='pointer-events: none;'>Import lib</h4>

```
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import math
```
<h4 style='pointer-events: none;'>Read data</h4>

```
data = pd.read_excel("Data/LSTM神经网络数据.xlsx", index_col=0)
target_column = "径流量"
features = data.drop(target_column, axis=1)
target = data[target_column]
```
<h4 style='pointer-events: none;'>Nomalize the data</h4>

```
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
features = x_scaler.fit_transform(features)
target = y_scaler.fit_transform(target.values.reshape(-1, 1))
```
<h4 style='pointer-events: none;'>Prepare input and output for the Transformer</h4>

```
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return (
            self.sequences[index],
            self.targets[index]
        )

def create_sequences(features, target, time_steps):
xs = []
ys = []

    for i in range(len(features) - time_steps):
        xs.append(features[i : i + time_steps])
        ys.append(target[i + time_steps])
    # convert data to numpy arrays
    xs = np.array(xs)
    ys = np.array(ys)
    # convert data to tensors
    xs = torch.from_numpy(xs).float()
    ys = torch.from_numpy(ys).float()
    return xs, ys

time_steps = 30
x, y = create_sequences(features, target, time_steps)
```
<h4 style='pointer-events: none;'>Split the dataset into training and testing sets</h4>

```
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1
x_train = x[:int(len(x) * train_ratio)]
y_train = y[:int(len(y) * train_ratio)]
x_valid = x[int(len(x) * train_ratio):int(len(x) * (train_ratio + valid_ratio))]
y_valid = y[int(len(y) * train_ratio):int(len(y) * (train_ratio + valid_ratio))]
x_test = x[int(len(x) * (train_ratio + valid_ratio)):]
y_test = y[int(len(y) * (train_ratio + valid_ratio)):]
```
<h4 style='pointer-events: none;'>Create Dataset and DataLoader</h4>

```
train_data = TimeSeriesDataset(x_train, y_train)
valid_data = TimeSeriesDataset(x_valid, y_valid)
test_data = TimeSeriesDataset(x_test, y_test)

batch_size = 128
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
```
<h4 style='pointer-events: none;'>Create s transformer model</h4>

```
class PositionalEncoding(nn.Module):
def __init__(self, d_model, max_len=5000):
super(PositionalEncoding, self).__init__()
pe = torch.zeros(max_len, d_model)
position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
pe = pe.unsqueeze(0).transpose(0, 1)
self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
def __init__(self, input_size, d_model, nhead, dim_feedforward, num_layers, output_size):
super(TransformerModel, self).__init__()
self.model_type = 'Transformer'
self.pos_encoder = PositionalEncoding(d_model)
self.encoder = nn.Linear(input_size, d_model)
self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers,
dim_feedforward, batch_first=True)
self.decoder = nn.Linear(d_model, output_size)

    def forward(self, src):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer(src, src)
        output = self.decoder(output[:, -1, :])
        return output

model = TransformerModel(
input_size=x_train.shape[-1],
d_model=512,
nhead=8,
dim_feedforward=2048,
num_layers=6,
output_size=1
)
```
<h4 style='pointer-events: none;'>loss function and optimizer</h4>

```
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
```
<h4 style='pointer-events: none;'>Train the model</h4>

```
epochs = 250
train_losses = []
valid_losses = []
for epoch in range(epochs):
batch_losses = []
for x_batch, y_batch in train_loader:
optimizer.zero_grad()
output = model(x_batch)
loss = criterion(output, y_batch)
loss.backward()
optimizer.step()
batch_losses.append(loss.item())
train_losses.append(np.mean(batch_losses))
print(f"Epoch {epoch} train loss: {train_losses[-1]}")

    with torch.no_grad():
        batch_losses = []
        for x_batch, y_batch in valid_loader:
            output = model(x_batch)
            loss = criterion(output, y_batch)
            batch_losses.append(loss.item())
        valid_losses.append(np.mean(batch_losses))
        # save the best model
        if epoch == 0 or valid_losses[-1] < min(valid_losses[:-1]):
            best_model = model.state_dict().copy()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} train loss: {train_losses[-1]} valid loss: {valid_losses[-1]}")
```
<h4 style='pointer-events: none;'>Plot loss</h4>

```
plt.plot(train_losses, label="Training loss")
plt.plot(valid_losses, label="Validation loss")
plt.legend()
plt.show()
```
<h4 style='pointer-events: none;'>Test the model</h4>

```
model.load_state_dict(best_model)
with torch.no_grad():
test_losses = []
for x_batch, y_batch in valid_loader:
output = model(x_batch)
loss = criterion(output, y_batch)
output = y_scaler.inverse_transform(output.numpy())
y_batch = y_scaler.inverse_transform(y_batch.numpy())
test_losses.append(loss.item())
print(f"Test loss: {np.mean(test_losses)}")
```
<h4 style='pointer-events: none;'>Predict on the entire test dataset</h4>

```
y_true = y_scaler.inverse_transform(y_test.numpy())
y_predict = model(x_test).numpy()
y_predict = y_scaler.inverse_transform(y_predict)
plt.plot(y_true, label="True")
plt.plot(y_predict, label="Predict")
plt.legend()
plt.show()
```



