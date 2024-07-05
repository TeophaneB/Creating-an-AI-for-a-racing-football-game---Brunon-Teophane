import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Neural network
class SoccerNet(nn.Module):
    def __init__(self, input_size):
        super(SoccerNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.layer3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.layer4 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(0.5)
        self.layer5 = nn.Linear(64, 32)
        self.dropout5 = nn.Dropout(0.5)
        self.output_layer = nn.Linear(32, 3)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu(self.layer1(x))
        x = self.dropout1(x)
        x = self.leaky_relu(self.layer2(x))
        x = self.dropout2(x)
        x = self.leaky_relu(self.layer3(x))
        x = self.dropout3(x)
        x = self.leaky_relu(self.layer4(x))
        x = self.dropout4(x)
        x = self.leaky_relu(self.layer5(x))
        x = self.dropout5(x)
        x = self.output_layer(x)
        return x

# - Training function
def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, num_epochs):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train() 
        total_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad() # clear the gradients
            outputs = model(inputs)
            labels = labels.squeeze(1)  # labels are correctly formatted
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses

# - Used for determining other evaluation metrics
def predict(model, dataloader): 
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)
            predictions.append(outputs)
    predictions = torch.cat(predictions, dim=0)
    return predictions.numpy()

# - Noise function
def add_noise(scaled_data, scales, means, noise_level=0.1):
    scales = torch.tensor(scales, dtype=torch.float32)
    means = torch.tensor(means, dtype=torch.float32)
    noise_std = scales * noise_level
    noise = torch.normal(means, noise_std) 
    return scaled_data + noise

# --- Dataset Loading / Preparation
path = "C:/Users/Teo/Documents/Computer Science/Y3/Game/new_ai_data.csv"
columns = ["k id",
           "k x", "k y", "k z", "k accel", "k steer", "k steer angle", "k speed", "k max speed", "k boost", "k brake", "k nitro",
           "b x", "b y", "b z", "b not moving",
           "closest k x", "closest k y", "closest k z",
           "blue t goal", "red t goal",
           "ball aim x", "ball aim y", "ball aim z"]
print("num of columns: ", len(columns))
data = pandas.read_csv(path)
data.columns = columns

team_0_data = data[data['k id'] == 0]
team_1_data = data[data['k id'] == 1]
target_columns = ["ball aim x", "ball aim y", "ball aim z"]
feature_columns = [col for col in columns if col not in ["k id"] + target_columns]

# - Extracting feature values
x_0 = team_0_data[feature_columns].values
x_1 = team_1_data[feature_columns].values
scaler = sklearn.preprocessing.StandardScaler()
X_scaled_0 = scaler.fit_transform(x_0)
mean_0 = scaler.mean_
scale_0 = scaler.scale_
X_scaled_1 = scaler.fit_transform(x_1)
mean_1 = scaler.mean_
scale_1 = scaler.scale_

# - Extract scaling parameters for STK
print(f"Team 0 - Mean: {mean_0}, Scale: {scale_0}")   # scale = scalar standard deviation
print(f"Team 1 - Mean: {mean_1}, Scale: {scale_1}")
with open("scaling_parameters_team_0.txt", "w") as file:
    file.write(f"Mean: {mean_0.tolist()}, Scale: {scale_0.tolist()}")
with open("scaling_parameters_team_1.txt", "w") as file:
    file.write(f"Mean: {mean_1.tolist()}, Scale: {scale_1.tolist()}")

# - Extracting target values (Loss fct requires [0.1] range thuse encoding labels)
y_0 = team_0_data[target_columns].values
y_1 = team_1_data[target_columns].values
label_encoder = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
all_targets = np.concatenate((y_0, y_1), axis=0)
label_encoder.fit(all_targets)
y_0_encoded = label_encoder.transform(y_0)
y_1_encoded = label_encoder.transform(y_1)

# - Extract scaling parameters for reversing the scaling
min_ = label_encoder.data_min_[0]  # Minimum value of the feature
max_ = label_encoder.data_max_[0]  # Maximum value of the feature
scale_ = label_encoder.scale_[0]  # Scale used by the MinMaxScaler
print(f"Min: {min_}, Max: {max_}, Scale: {scale_}")
with open("scaling_parameters.txt", "w") as file:
    file.write(f"Min: {min_}, Max: {max_}, Scale: {scale_}")

# - Convert the data to PyTorch tensors
X_tensor_0 = torch.tensor(X_scaled_0, dtype=torch.float32)
X_tensor_1 = torch.tensor(X_scaled_1, dtype=torch.float32)
y_tensor_0 = torch.tensor(y_0_encoded, dtype=torch.float32)
y_tensor_1 = torch.tensor(y_1_encoded, dtype=torch.float32)

print(f"Shape of X_tensor: {X_tensor_0.shape[1]}")
print(f"Shape of Y_tensor: {y_tensor_1.shape[1]}")
print(f"X_tensor[1]: {X_tensor_0[1]}")
print(f"Y_tensor[1]: {y_tensor_1[1]}")

# --- Training parameters
model_0 = SoccerNet(X_tensor_0.shape[1])
model_1 = SoccerNet(X_tensor_1.shape[1])

criterion = nn.MSELoss()
optimizer_0 = optim.Adam(model_0.parameters(), lr=0.000001)
optimizer_1 = optim.Adam(model_1.parameters(), lr=0.000001)

scheduler_0 = ReduceLROnPlateau(optimizer_0, mode='min', factor=0.1, patience=10, verbose=True)
scheduler_1 = ReduceLROnPlateau(optimizer_1, mode='min', factor=0.1, patience=10, verbose=True)

num_epochs = 500

X_train_0, X_val_0, Y_train_0, Y_val_0 = sklearn.model_selection.train_test_split(X_tensor_0, y_tensor_0, test_size=0.2, random_state=42)
X_train_1, X_val_1, Y_train_1, Y_val_1 = sklearn.model_selection.train_test_split(X_tensor_1, y_tensor_1, test_size=0.2, random_state=42)

print(f"X_train_0[1]: {X_train_0[1]}")
X_train_0_noise = add_noise(X_train_0, scale_0, mean_0)
print(f"Noise X_train_0[1]: {X_train_0_noise[1]}")
X_train_1_noise = add_noise(X_train_1, scale_1, mean_1)

# - Plot histograms of before and after adding noise
X_train_0_numpy = X_train_0.numpy()
X_train_0_noise_numpy = X_train_0_noise.numpy()
X_train_1_numpy = X_train_1.numpy()
X_train_1_noise_numpy = X_train_1_noise.numpy()

plt.figure(figsize=(4, 24))  # Increased height for more vertical space

# Plot before adding noise
plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
plt.hist(X_train_0_numpy[0:30], bins=10, alpha=0.7, label='Before Noise', rwidth=0.8)
plt.title('Histogram of Row Before Noise')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Plot after adding noise
plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
plt.hist(X_train_0_noise_numpy[0:30], bins=10, alpha=0.7, label='After Noise', rwidth=0.8)
plt.title('Histogram of Row After Noise')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# - Create data loaders for each team
train_dataset_0 = TensorDataset(X_train_0_noise, Y_train_0)
train_loader_0 = DataLoader(train_dataset_0, batch_size=64, shuffle=True)
val_dataset_0 = TensorDataset(X_val_0, Y_val_0)
val_loader_0 = DataLoader(val_dataset_0, batch_size=64, shuffle=False)

train_dataset_1 = TensorDataset(X_train_1_noise, Y_train_1)
train_loader_1 = DataLoader(train_dataset_1, batch_size=64, shuffle=True)
val_dataset_1 = TensorDataset(X_val_1, Y_val_1)
val_loader_1 = DataLoader(val_dataset_1, batch_size=64, shuffle=False)

# --- Train the models
train_losses_0, val_losses_0 = train_model(train_loader_0, val_loader_0, model_0, criterion, optimizer_0, scheduler_0, num_epochs)
train_losses_1, val_losses_1 = train_model(train_loader_1, val_loader_1, model_1, criterion, optimizer_1, scheduler_1, num_epochs)

# Save the trained models
model_0.eval()
model_1.eval()

# Make predictions on the validation set
y_pred_0 = predict(model_0, val_loader_0)
y_pred_1 = predict(model_1, val_loader_1)

# - Calculate metrics for model_0
mae_0 = mean_absolute_error(Y_val_0.numpy(), y_pred_0)
mse_0 = mean_squared_error(Y_val_0.numpy(), y_pred_0)
rmse_0 = mean_squared_error(Y_val_0.numpy(), y_pred_0, squared=False)
r2_0 = r2_score(Y_val_0.numpy(), y_pred_0)

# - Calculate metrics for model_1
mae_1 = mean_absolute_error(Y_val_1.numpy(), y_pred_1)
mse_1 = mean_squared_error(Y_val_1.numpy(), y_pred_1)
rmse_1 = mean_squared_error(Y_val_1.numpy(), y_pred_1, squared=False)
r2_1 = r2_score(Y_val_1.numpy(), y_pred_1)

# Print the metrics
print(f"Model 0 - MAE: {mae_0}, MSE: {mse_0}, RMSE: {rmse_0}, R2: {r2_0}")
print(f"Model 1 - MAE: {mae_1}, MSE: {mse_1}, RMSE: {rmse_1}, R2: {r2_1}")

# - Save the models
example_input_0 = torch.rand(1, X_train_0_noise.shape[1])  # random example inputs to save the models
example_input_1 = torch.rand(1, X_train_1_noise.shape[1])
traced_script_module_0 = torch.jit.trace(model_0, example_input_0)
traced_script_module_1 = torch.jit.trace(model_1, example_input_1)
traced_script_module_0.save("models/net_0_diff.pt")
traced_script_module_1.save("models/net_1_diff.pt")

# - Plot the training and validation losses
def plot_losses(train_losses_0, val_loss_0, train_losses_1, val_loss_1):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns of plots

    # Plot for the first set of losses
    axs[0].plot(train_losses_0, label='Training Loss 0')
    axs[0].plot(val_loss_0, label='Validation Loss 0')
    axs[0].set_title('Training and Validation Losses for Model 0')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot for the second set of losses
    axs[1].plot(train_losses_1, label='Training Loss 1')
    axs[1].plot(val_loss_1, label='Validation Loss 1')
    axs[1].set_title('Training and Validation Losses for Model 1')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

plot_losses(train_losses_0, val_losses_0, train_losses_1, val_losses_1)