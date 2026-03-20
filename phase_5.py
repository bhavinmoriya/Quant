"""
Phase 5: Advanced Alpha Generation (LSTM Sequence Modeling)
------------------------------------------------------------
Goal:
1. Transition from tabular data (XGBoost) to sequential 3D data (PyTorch LSTMs).
2. Because real-time (RT) electricity prices update every 5 minutes and cascade
   like a physical wave across the grid, LSTM memory cells can "see" momentum
   better than independent rows.
3. Build a lightweight PyTorch training loop optimized for Kaggle (8GB RAM).
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# Plotting configuration
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (14, 6)

# Set PyTorch seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1. Simulate 5-Minute Grid Telemetry ---
def get_sequential_data(days: int = 15) -> pl.DataFrame:
    print(f"[*] Simulating {days} days of 5-minute Real-Time (RT) Pricing...")

    # 5-minute intervals = 12 per hour * 24 = 288 per day
    intervals_per_day = 288
    total_intervals = days * intervals_per_day

    timestamps = [datetime(2023, 8, 1) + timedelta(minutes=5*i) for i in range(total_intervals)]

    # Base daily curve
    base_price = 30.0 + 15 * np.sin(np.linspace(0, days * 2 * np.pi, total_intervals))
    rt_prices = base_price + np.random.normal(0, 2, total_intervals)

    # Simulate sequential cascading momentum (A generator trip causes prices to spike across multiple 5-min intervals)
    # Using an auto-regressive process to simulate "memory" in the price
    for i in range(1, total_intervals):
        # 30% of the previous interval's chaos carries over to the current
        rt_prices[i] += 0.30 * (rt_prices[i-1] - base_price[i-1])

    # Inject a severe grid stress event
    event_start = 288 * 5 + 144  # Day 5, noon
    rt_prices[event_start:event_start+12] += np.linspace(50, 250, 12) # Cascading 1-hour spike
    rt_prices[event_start+12:event_start+24] += np.linspace(250, 0, 12) # Recovery

    df = pl.DataFrame({
        "timestamp": timestamps,
        "rt_lmp": rt_prices
    })

    return df

# --- 2. PyTorch Data Engineering (2D Tabular -> 3D Sequential) ---
def create_sequences(data: np.ndarray, seq_length: int):
    """
    Transforms [1000, 1] data into [1000, seq_length, 1] tensors.
    If seq_len=12 (1 hour of 5-min intervals), the model looks at the past hour to predict the next 5 mins.
    """
    xs = []
    ys = []

    # We predict the exact next step (t+1) based on the past `seq_length` steps
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

# --- 3. The LSTM Network Architecture ---
class EnergyLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, output_size=1):
        super(EnergyLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        # The LSTM Layer
        # batch_first=True expects input shape: [batch_size, sequence_length, features]
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=1, batch_first=True)

        # Readout completely connected layer
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # lstm_out contains the output features from the last layer of the LSTM for each timestep
        # We only care about the very last timestep's hidden state to make our final prediction
        lstm_out, (hidden, cell) = self.lstm(input_seq)

        # Get the output from the last timestep: lstm_out[:, -1, :]
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# --- 4. Training Loop ---
def train_model(model, train_loader, epochs=15):
    criterion = nn.MSELoss()
    # AdamW is generally preferred over standard Adam for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)

    print("\n[*] Training LSTM (Memory Optimized for Kaggle)...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for seq, labels in train_loader:
            optimizer.zero_grad()

            y_pred = model(seq)
            loss = criterion(y_pred, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch+1) % 5 == 0:
            print(f"    Epoch {epoch+1:2d}/{epochs} | MSE Loss: {total_loss/len(train_loader):.4f}")

def main():
    print("=== Algorithmic Trading Phase 5: Deep Learning Momentum (LSTM) ===")

    # 1. Fetch 5-Minute Data
    df = get_sequential_data(days=15)

    # 2. Scale the data (LSTMs perform terribly on unscaled data)
    print("[*] Normalizing data with MinMaxScaler (Required for Neural Networks)")
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # We only use price history for this model. (Univariate sequence)
    price_data = df["rt_lmp"].to_numpy().reshape(-1, 1)
    scaled_data = scaler.fit_transform(price_data)

    # 3. Create 3D Sequences
    SEQUENCE_LENGTH = 12 # Look back 1 hour (12 * 5mins) to predict next 5 mins
    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)

    # Split Train/Test (first 10 days train, last 5 days test)
    intervals_per_day = 288
    train_size = 10 * intervals_per_day

    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader to manage RAM usage in batches
    # Batch size of 64 is safe for 8GB RAM Kaggle limits
    train_data = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=False) # NEVER shuffle time series sequences!

    # 4. Initialize and Train Model
    model = EnergyLSTM(input_size=1, hidden_layer_size=32, output_size=1)
    train_model(model, train_loader, epochs=15)

    # 5. Evaluate Out-Of-Sample
    print("\n[*] Evaluating Out-of-Sample Predictions...")
    model.eval()
    with torch.no_grad():
        test_predictions_scaled = model(X_test_t).numpy()

    # Inverse transform back to real dollar values
    test_predictions = scaler.inverse_transform(test_predictions_scaled)
    actual_prices = scaler.inverse_transform(y_test)

    # 6. Plotting
    print("[*] Plotting Sequential Edge vs Actual Cascade...")

    # Grab the corresponding timestamps for the test subset
    test_timestamps = df["timestamp"].to_list()[train_size + SEQUENCE_LENGTH:]

    # We'll zoom in on the last 2 days of testing to see the actual 5-min ticks
    zoom_len = 288 * 2

    plt.figure(figsize=(15, 8))
    plt.plot(test_timestamps[-zoom_len:], actual_prices[-zoom_len:], label="Actual RT 5-Min Price", color="black", alpha=0.6)
    plt.plot(test_timestamps[-zoom_len:], test_predictions[-zoom_len:], label="LSTM T+1 Forecast", color="red", linestyle="--", linewidth=2)

    # The LSTM excels at catching the "start of the cascade" because it remembers the prior 12 sequence ticks
    plt.title("PyTorch LSTM: Capturing 5-Minute Grid Momentum", fontsize=15)
    plt.ylabel("Real-Time LMP ($/MWh)")
    plt.xlabel("Datetime")
    plt.legend()

    plt.tight_layout()
    plt.savefig("phase_5_output.png")
    print("[+] Plot saved as 'phase_5_output.png'. You can view it in your Kaggle output directory.")

if __name__ == "__main__":
    main()
