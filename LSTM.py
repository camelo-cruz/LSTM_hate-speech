import os 
import torch
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from dataset import TextDataset
from transformers import BertTokenizer, BertModel
from utils import preprocess_data

current_dir = os.getcwd()
csv_path = os.path.join(current_dir, 'data', 'final_hateXplain.csv')

df = preprocess_data(csv_path)

text = df.loc['comment']
labels = df.loc['label'] 
context_columns = df.columns.difference(['comment', 'label'])




#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#dataset = TextDataset(labels="path/to/labels.csv", text="path/to/text.csv", tokenizer=tokenizer.tokenize)

#train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
#test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)