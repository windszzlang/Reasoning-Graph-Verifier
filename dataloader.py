import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data, is_test=False):
        self.is_test = is_test
        if not self.is_test:
            self.data = data
        else:
            self.data = data

    def __getitem__(self, idx):
        assert idx < len(self.data)
        if not self.is_test:
            return self.data[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


class MyCollator():
    def __init__(self, max_len, tokenizer, device, is_test=False):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.device = device
        self.is_test = is_test

    def __call__(self, batch_data):
        return batch_data


def get_dataloader(data, batch_size, max_len, tokenizer, device, is_shuffle=False, is_test=False):
    collate_fn = MyCollator(max_len, tokenizer, device, is_test)
    dataset = MyDataset(data, is_test)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_shuffle,
        collate_fn=collate_fn
    )
    return dataloader