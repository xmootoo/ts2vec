import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import TensorDataset, Dataset, DataLoader

class ForecastingDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, input_dims=None):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dims = input_dims

    def __len__(self):
        return self.data.shape[1] - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        input_data = self.data[:, idx:idx+self.seq_len, :]
        if self.input_dims:
            target_data = self.data[:, idx+self.seq_len:idx+self.seq_len+self.pred_len, self.input_dims:]
        else:
            target_data = self.data[:, idx+self.seq_len:idx+self.seq_len+self.pred_len, -1].unsqueeze(-1)
        return input_data, target_data

def get_forecast_loader(data, seq_len, pred_len, batch_size, input_dims=None, shuffle=True):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    dataset = ForecastingDataset(data, seq_len, pred_len, input_dims)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)

class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

def get_downstream_ett_model(model, embed_dim, num_channels, seq_len, pred_len, batch_size):

    downstream_model = nn.Sequential(Squeeze(1), # (B, L, input_dim)
                                    model, # (B, L, embed_dim)
                                    Reshape((batch_size, seq_len*embed_dim)), # (B, L*embed_dim),
                                    nn.Linear(seq_len*embed_dim, num_channels*pred_len), # (L*embed_dim) -> (num_channels*pred_len)
                                    View((batch_size, pred_len, num_channels))) # (B, pred_len, num_channels)
    # Count parameters
    num_params = sum(p.numel() for p in downstream_model.parameters())
    print(f"Downstream Model with {num_params} parameters loaded")

    return downstream_model

def get_downstream_model(model, embed_dim, num_channels, seq_len, pred_len, batch_size):

    downstream_model = nn.Sequential(Reshape((batch_size*num_channels, seq_len, -1)), # (B*M, L, input_dim)
                                    model, # (B*M, L, embed_dim)
                                    Reshape((batch_size*num_channels, seq_len*embed_dim)), # (B*M, L*embed_dim),
                                    nn.Linear(seq_len*embed_dim, pred_len), # (L*embed_dim) -> (T)
                                    View((batch_size, num_channels, pred_len))) # (B, M, T)

    # Count parameters
    num_params = sum(p.numel() for p in downstream_model.parameters())
    print(f"Downstream Model with {num_params} parameters loaded")

    return downstream_model


def fine_tune(pretrained_model,
              data,
              name,
              train_slice,
              valid_slice,
              test_slice,
              lr=1e-4,
              epochs=10,
              optimizer=torch.optim.Adam,
              criterion=nn.MSELoss,
              seq_len=512,
              pred_lens=[96, 192, 336, 720],
              batch_size=32,
              device=None,
              embed_dim=320,
              mae=True,
              input_dims=None):

    train_data = data[:, train_slice]
    valid_data = data[:, valid_slice]
    test_data = data[:, test_slice]


    num_channels = train_data.shape[0]

    criterion = criterion()

    if mae:
        mae_loss = torch.nn.L1Loss()
        test_mae = 0

    results = {}

    # If dir does not exist make it
    if not os.path.exists(f'downstream/{name}'):
        os.makedirs(f'downstream/{name}', exist_ok=True)

    for pred_len in pred_lens:

        best_model_path = f'downstream/{name}/{pred_len}.pth'

        if "ETT" in name:
            model = get_downstream_ett_model(pretrained_model, embed_dim, num_channels, seq_len, pred_len, batch_size).to(device)
            k = 1
        else:
            model = get_downstream_model(pretrained_model, embed_dim, num_channels, seq_len, pred_len, batch_size).to(device)
            k = 3

        optimizer = optimizer(model.parameters(), lr=lr)


        train_loader = get_forecast_loader(train_data, seq_len, pred_len, batch_size, input_dims, shuffle=True)
        val_loader = get_forecast_loader(valid_data, seq_len, pred_len, batch_size, input_dims, shuffle=True)
        test_loader = get_forecast_loader(test_data, seq_len, pred_len, batch_size, input_dims, shuffle=False)

        best_val_loss = float('inf')

        #<-----Training---->
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            print(f"Training ({epoch}/{epochs})")
            for i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()

                x = x.to(device)
                y = y.squeeze(k).to(device)

                y_hat = model(x)

                loss = criterion(y_hat, y)
                train_loss+=loss.item()
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print(f"({i}/{len(train_loader)}) | loss: {loss.item()}")

            train_loss/=len(train_loader)
            print(f"Epoch {epoch} Train Loss: {train_loss}")

            #<-----Validation---->
            print(f"Validating ({epoch}/{epochs})")
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, (x, y) in enumerate(val_loader):
                    x = x.to(device)
                    y = y.squeeze(k).to(device)

                    y_hat = model(x)

                    loss = criterion(y_hat, y)
                    val_loss+= loss.item()

                    if i%100==0:
                        print(f"({i}/{len(val_loader)}) | loss: {loss.item()}")


            val_loss/=len(val_loader)
            print(f"Epoch {epoch} Validation Loss: {val_loss}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Saving best model")
                torch.save(model, best_model_path)


        #<-----Testing---->
        print(f"Testing... Loading best model.")
        model = torch.load(best_model_path) # Load best model
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.squeeze(k).to(device)

                y_hat = model(x)

                test_loss += criterion(y_hat, y).item()

                if mae:
                    test_mae+= mae_loss(y_hat, y).item()


        print(f"Test MSE: {test_loss/len(test_loader)} for pred_len: {pred_len}")
        print(f"Test MAE: {test_mae/len(test_loader)} for pred_len: {pred_len}")

        results[pred_len] = (test_loss/len(test_loader), test_mae/len(test_loader))

    return results
