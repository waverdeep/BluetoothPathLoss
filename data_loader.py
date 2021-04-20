import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class PathLossDataset(Dataset):
    def __init__(self, input_dir):
        self.dataset = pd.read_csv(input_dir)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        line = self.dataset.loc[idx]
        line = line.tolist()
        del line[0]
        y_data = line[0]
        del line[0]
        x_data = line
        x_data = torch.tensor(x_data, dtype=torch.float)
        y_data = torch.tensor(y_data, dtype=torch.float)
        return x_data, y_data


def load_pathloss_dataset(input_dir, batch_size, shuffle, num_workers):
    pathloss_dataset = PathLossDataset(input_dir=input_dir)
    pathloss_dataloader = DataLoader(pathloss_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return pathloss_dataloader

