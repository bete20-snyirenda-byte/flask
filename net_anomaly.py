import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# ===============================
# 1. Load and preprocess data
# ===============================
file_path = "C:/Users/j/Desktop/traffic_stats_combined1.xlsx"
df = pd.read_excel(file_path)

# Ensure 'time' column exists
if "time" not in df.columns:
    raise ValueError("No 'time' column found in Excel file")

# Convert to datetime
df['time'] = pd.to_datetime(df['time'], errors='coerce', dayfirst=True)
df = df.dropna(subset=['time'])

# Extract features from time
df['day'] = df['time'].dt.dayofweek + 1   # 1=Monday … 7=Sunday
df['hour'] = df['time'].dt.hour

print("time dtype:", df['time'].dtype)
print(df[['time', 'day', 'hour']].head())

# Select numeric columns + engineered features
numeric_features = df.select_dtypes(include=[np.number])
numeric_features = pd.concat([numeric_features, df[['day', 'hour']]], axis=1)

# Handle NaN/inf
numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
numeric_features = numeric_features.fillna(method='ffill').fillna(method='bfill')

# ===============================
# 2. Scale data
# ===============================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(numeric_features)

# ===============================
# 3. Create sequences
# ===============================
SEQ_LEN = 144  # ≈ one day if 10min intervals
def create_sequences(data, seq_length=SEQ_LEN):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

X = create_sequences(scaled_data, SEQ_LEN)
print("Sequence shape:", X.shape)

# Torch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)

# ===============================
# 4. Define LSTM Autoencoder
# ===============================
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_size=64):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, n_features, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden = hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2)  # repeat for each timestep
        reconstructed, _ = self.decoder(hidden)
        return reconstructed

n_features = X.shape[2]
model = LSTMAutoencoder(n_features)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ===============================
# 5. Train model
# ===============================
EPOCHS = 10
BATCH_SIZE = 32

dataset = torch.utils.data.DataLoader(X_tensor, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    for batch in dataset:
        reconstructed = model(batch)
        loss = criterion(reconstructed, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

# ===============================
# 6. Detect anomalies
# ===============================
model.eval()
with torch.no_grad():
    reconstructed = model(X_tensor)
    mse = torch.mean((X_tensor - reconstructed) ** 2, dim=(1, 2))  # per sequence

threshold = np.percentile(mse.numpy(), 99)  # top 1% as anomalies
anomalies = (mse.numpy() > threshold).astype(int)
print(f"Anomalies detected: {anomalies.sum()}")

# ===============================
# 7. Report anomaly times
# ===============================
time_info = df[['time', 'day', 'hour']].iloc[SEQ_LEN:]  # align with sequences
anomalous_times = time_info[anomalies == 1]
print("Anomalous timestamps:\n", anomalous_times.head())

# ===============================
# 8. Plot anomalies in n_bytes
# ===============================
if "n_bytes" not in df.columns:
    raise ValueError("No 'n_bytes' column found in Excel file for plotting anomalies")

time_index = df['time'].iloc[SEQ_LEN:]
n_bytes_values = df["n_bytes"].values[SEQ_LEN:]
anomaly_points = anomalies == 1

plt.figure(figsize=(15, 6))
plt.plot(time_index, n_bytes_values, label="n_bytes", color="blue")
plt.scatter(time_index[anomaly_points], n_bytes_values[anomaly_points],
            color="red", marker="x", label="Anomalies")
plt.title("Anomalies in n_bytes Over Time")
plt.xlabel("Time")
plt.ylabel("n_bytes")
plt.legend()
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# ===============================
# 1. Load and preprocess data
# ===============================
file_path = "C:/Users/j/Desktop/traffic_stats_combined1.xlsx"
df = pd.read_excel(file_path)

# Ensure 'time' column exists
if "time" not in df.columns:
    raise ValueError("No 'time' column found in Excel file")

# Convert to datetime
df['time'] = pd.to_datetime(df['time'], errors='coerce', dayfirst=True)
df = df.dropna(subset=['time'])

# Extract features from time
df['day'] = df['time'].dt.dayofweek + 1   # 1=Monday … 7=Sunday
df['hour'] = df['time'].dt.hour

print("time dtype:", df['time'].dtype)
print(df[['time', 'day', 'hour']].head())

# Select numeric columns + engineered features
numeric_features = df.select_dtypes(include=[np.number])
numeric_features = pd.concat([numeric_features, df[['day', 'hour']]], axis=1)

# Handle NaN/inf
numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
numeric_features = numeric_features.fillna(method='ffill').fillna(method='bfill')

# ===============================
# 2. Scale data
# ===============================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(numeric_features)

# ===============================
# 3. Create sequences
# ===============================
SEQ_LEN = 144  # ≈ one day if 10min intervals
def create_sequences(data, seq_length=SEQ_LEN):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

X = create_sequences(scaled_data, SEQ_LEN)
print("Sequence shape:", X.shape)

# Torch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)

# ===============================
# 4. Define LSTM Autoencoder
# ===============================
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_size=64):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, n_features, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden = hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2)  # repeat for each timestep
        reconstructed, _ = self.decoder(hidden)
        return reconstructed

n_features = X.shape[2]
model = LSTMAutoencoder(n_features)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ===============================
# 5. Train model
# ===============================
EPOCHS = 10
BATCH_SIZE = 32

dataset = torch.utils.data.DataLoader(X_tensor, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    for batch in dataset:
        reconstructed = model(batch)
        loss = criterion(reconstructed, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

# ===============================
# 6. Detect anomalies
# ===============================
model.eval()
with torch.no_grad():
    reconstructed = model(X_tensor)
    mse = torch.mean((X_tensor - reconstructed) ** 2, dim=(1, 2))  # per sequence

threshold = np.percentile(mse.numpy(), 99)  # top 1% as anomalies
anomalies = (mse.numpy() > threshold).astype(int)
print(f"Anomalies detected: {anomalies.sum()}")

# ===============================
# 7. Report anomaly times
# ===============================
time_info = df[['time', 'day', 'hour']].iloc[SEQ_LEN:]  # align with sequences
anomalous_times = time_info[anomalies == 1]
print("Anomalous timestamps:\n", anomalous_times.head())

# ===============================
# 8. Plot anomalies in n_bytes
# ===============================
if "n_bytes" not in df.columns:
    raise ValueError("No 'n_bytes' column found in Excel file for plotting anomalies")

time_index = df['time'].iloc[SEQ_LEN:]
n_bytes_values = df["n_bytes"].values[SEQ_LEN:]
anomaly_points = anomalies == 1

plt.figure(figsize=(15, 6))
plt.plot(time_index, n_bytes_values, label="n_bytes", color="blue")
plt.scatter(time_index[anomaly_points], n_bytes_values[anomaly_points],
            color="red", marker="x", label="Anomalies")
plt.title("Anomalies in n_bytes Over Time")
plt.xlabel("Time")
plt.ylabel("n_bytes")
plt.legend()
plt.show()