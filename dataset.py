from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer

class TextDataset(Dataset):
    def __init__(self, text, labels):
        self.text = text.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)

        assert len(self.text) == len(self.labels)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text.iloc[idx]
        label = self.labels.iloc[idx]

        return text, label
