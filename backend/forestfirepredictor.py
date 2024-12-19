# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi, Repository, login

# %% [markdown]
# # Creating a Model class that inherits the parent nn.Module

# %%
class Model(nn.Module):
    def __init__(self, in_features=6, h1=8, h2=9, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

# %%
torch.manual_seed(42)
model = Model()

# %%
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline


# %%
dataFile = '/kaggle/input/forest-fire-data/forestFire.csv'
my_df = pd.read_csv(dataFile)
# Remove rows with missing values
my_df.dropna(inplace=True)


# %%
my_df

# %%
X = my_df.drop('fire', axis=1)
y = my_df['fire']

# %% [markdown]
# # Convert DataFrames to NumPy arrays

# %%
X = X.values
y = y.values

# %%
from sklearn.model_selection import train_test_split

# %% [markdown]
# # Train Test Split

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# # Convert to Tensors

# %%
# Convert X_train to a numeric data type
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# Convert to PyTorch tensor
X_train = torch.FloatTensor(X_train)

X_test = torch.FloatTensor(X_test)

# %%
# Convert to PyTorch tensor
y_train = torch.LongTensor(y_train)

y_test = torch.LongTensor(y_test)

# %% [markdown]
# # Set the criterion of model to measure the error

# %%
criterion = nn.CrossEntropyLoss()

# %% [markdown]
# **Choose Adam optimizer, lr = learning rate (if error does not go down after epochs, lower our learning rate)**

# %%
optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)

# %% [markdown]
# # Train Model
# Choose number of epochs

# %%
epochs = 100
losses = []

for i in range(epochs):
    # Go forward to get prediction
    y_pred = model.forward(X_train) # Get predicted results

    # Measure the loss
    loss = criterion(y_pred, y_train) # Predicted value vs trained

    # Keep track of losses
    losses.append(loss.detach().numpy())

    # Print every 10 epochs
    if(i % 10 == 0):
        print(f'Epoch {i} and loss : {loss}')

    # Do back propagation, take the error rate and forward propagation and feed 
    # it back through the network to fine tune the weights

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# %% [markdown]
# # Graph the results

# %%
plt.plot(range(epochs), losses)
plt.ylabel("error")
plt.xlabel("epoch")

# %% [markdown]
# # Evaluate Model on Test Dataset

# %%
with torch.no_grad(): # Turn off back propagation
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test) # Find error    

# %%
loss

# %%
correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        # Will tell if fire or not fire
        print(f'({i+1}) {str(y_val)} Prediction: {y_val.argmax().item()} \t Value: {y_test[i]}')

        # Correct or not
        if (y_val.argmax().item() == y_test[i]):
            correct += 1

print(f'We got {correct} correct!')

# %% [markdown]
# # Testing model on new data

# %%
newFireLocation = torch.tensor([12, -5, 20, 15, 0.1, 2])

with torch.no_grad():
    raw_output = model(newFireLocation)  # This gives the raw output (logits)
    print(f'Raw output: {raw_output}')

    # Convert logits to probabilities
    probabilities = torch.softmax(raw_output, dim=0)

    # Convert probabilities to percentages
    percentages = probabilities * 100
    print(f'Predicted probabilities as percentages: {percentages}')

    # If you want the predicted class label (index of highest probability)
    predicted_class = probabilities.argmax().item()
    print(f'Predicted class: {predicted_class}')



