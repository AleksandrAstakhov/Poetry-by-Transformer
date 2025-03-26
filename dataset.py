from torch.utils.data import Dataset

import torch

class PoemDataset(Dataset):

    def __init__(self, data, tokenizer, block_size):

        self.data = self.tokens = tokenizer(
            data, truncation=False, padding=False)["input_ids"]
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, index):
        return torch.tensor(self.data[index: index + self.block_size]), torch.tensor(self.data[index + 1: index + self.block_size + 1])
