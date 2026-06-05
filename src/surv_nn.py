import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional

class SurvivalDataset(Dataset):
    def __init__(self, X: np.ndarray, durations: np.ndarray, events: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.durations = torch.tensor(durations, dtype=torch.float32)
        self.events = torch.tensor(events, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.durations[idx], self.events[idx]

class DeepSurvNet(nn.Module):
    def __init__(self, in_features: int, hidden_dims: List[int] = [64, 64, 32], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = in_features
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

def cox_partial_likelihood_loss(log_hazard: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    # duration 내림차순 정렬 (Cox PH 효율적 계산을 위함)
    order = torch.argsort(durations, descending=True)
    log_hazard = log_hazard[order]
    events = events[order]

    # log-sum-exp trick (numerical stability)
    log_cumsum_hazard = torch.logcumsumexp(log_hazard, dim=0)
    uncensored_loss = (log_hazard - log_cumsum_hazard) * events
    n_events = events.sum()

    if n_events == 0:
        return torch.tensor(0.0, requires_grad=True)

    return -uncensored_loss.sum() / n_events

class DeepSurvModel:
    def __init__(self, in_features: int, hidden_dims: List[int] = [64, 64, 32], dropout: float = 0.3, lr: float = 1e-3, weight_decay: float = 1e-4, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = DeepSurvNet(in_features, hidden_dims, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=5, factor=0.5)
        self.train_losses = []
        self.val_losses = []

    def fit(self, X_train: np.ndarray, dur_train: np.ndarray, evt_train: np.ndarray, X_val: np.ndarray, dur_val: np.ndarray, evt_val: np.ndarray, epochs: int = 100, batch_size: int = 512, patience: int = 15, verbose: bool = True) -> "DeepSurvModel":
        train_loader = DataLoader(SurvivalDataset(X_train, dur_train, evt_train), batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(SurvivalDataset(X_val, dur_val, evt_val), batch_size=batch_size, shuffle=False)

        best_val = float("inf")
        best_state = None
        patience_cnt = 0

        for epoch in range(1, epochs + 1):
            self.net.train()
            tr_loss = 0.0
            for Xb, tb, eb in train_loader:
                Xb, tb, eb = Xb.to(self.device), tb.to(self.device), eb.to(self.device)
                self.optimizer.zero_grad()
                loss = cox_partial_likelihood_loss(self.net(Xb), tb, eb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()
                tr_loss += loss.item()
            
            tr_loss /= len(train_loader)

            self.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xb, tb, eb in val_loader:
                    Xb, tb, eb = Xb.to(self.device), tb.to(self.device), eb.to(self.device)
                    val_loss += cox_partial_likelihood_loss(self.net(Xb), tb, eb).item()
            val_loss /= max(len(val_loader), 1)

            self.train_losses.append(tr_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience: break

        if best_state: self.net.load_state_dict(best_state)
        return self

    def predict_log_hazard(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.net(X_t).cpu().numpy()

    def predict_expected_duration(self, X: np.ndarray, max_duration: float = 545.0) -> np.ndarray:
        log_h = self.predict_log_hazard(X)
        hazard = np.exp(log_h)
        hazard_norm = (hazard - hazard.min()) / (hazard.max() - hazard.min() + 1e-8)
        return (max_duration * (1 - hazard_norm) + 1.0).astype(np.float32)

    def save(self, path: str): torch.save(self.net.state_dict(), path)
    def load(self, path: str): self.net.load_state_dict(torch.load(path, map_location=self.device))