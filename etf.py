import yfinance as yf
import pandas as pd
import numpy as np
import ta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Parameters
ticker = "SPY"
start = "2010-01-01"
end = "2023-12-31"
window = 60
forecast_horizon = 5  # 预测未来5天收益

print("Fetching data...")
df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

print("Adding indicators...")
close = df['Close']
if isinstance(close, pd.DataFrame):
    close = close.squeeze()

# 技术指标
df['return'] = close.pct_change()
df['rsi'] = ta.momentum.RSIIndicator(close).rsi()
macd = ta.trend.MACD(close)
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
bbands = ta.volatility.BollingerBands(close)
df['bb_high'] = bbands.bollinger_hband()
df['bb_low'] = bbands.bollinger_lband()

# Drop NaN
df = df.dropna()

# 目标: 未来5天累计收益
df['future_return'] = df['return'].rolling(forecast_horizon).sum().shift(-forecast_horizon)
df = df.dropna()

# 特征列表
features = ['return', 'rsi', 'macd', 'macd_signal', 'bb_high', 'bb_low']

# 特征标准化
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# 构建 LSTM 输入输出
X, y = [], []
for i in range(len(df) - window - forecast_horizon + 1):
    X.append(df[features].iloc[i:i+window].values)
    y.append(df['future_return'].iloc[i+window-1])
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# LSTM模型
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = LSTMRegressor(input_dim=len(features), hidden_dim=32)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training...")
for epoch in range(20):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb).squeeze()
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss={total_loss/len(train_loader):.6f}")

print("Evaluating...")
model.eval()
y_pred, y_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        outputs = model(xb).squeeze()
        y_pred.extend(outputs.numpy())
        y_true.extend(yb.numpy())

plt.figure(figsize=(10,5))
plt.plot(y_true[:200], label="True Future Return")
plt.plot(y_pred[:200], label="Predicted")
plt.legend()
plt.show()
