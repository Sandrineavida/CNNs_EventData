from torch.utils.data import DataLoader
import torch

def get_dataloaders(train_data_path, valid_data_path, test_data_path, batch_size=32):
    train_dataset = torch.load(train_data_path)
    valid_dataset = torch.load(valid_data_path)
    test_dataset = torch.load(test_data_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, valid_dataloader, test_dataloader