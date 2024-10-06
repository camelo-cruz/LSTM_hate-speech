import os
import torch
import preprocessing
import random
import numpy as np
from tqdm import tqdm
from itertools import product
from model import LSTM
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TextDataset

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_dir = os.getcwd()
csv_path = os.path.join(current_dir, 'data', 'final_hateXplain.csv')


train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

learning_rates = [1e-3, 1e-4, 1e-5]
batch_sizes = [16, 32, 64]
hidden_sizes = [60, 128, 256]
num_layers_list = [2, 3, 4 ,5]
num_epochs = 50

param_combinations = list(product(learning_rates, batch_sizes, hidden_sizes, num_layers_list))

best_loss = float('inf')
run = 0
best_params = None

for params in param_combinations:
    run += 1
    learning_rate, batch_size, hidden_size, num_layers = params

    train_sample, test_sample = preprocessing.preprocess_data(csv_path)
    train_dataset = TextDataset(train_sample['input_ids'], train_sample['mask'], train_sample['labels'])
    test_dataset = TextDataset(test_sample['input_ids'], test_sample['mask'], test_sample['labels'])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    model = LSTM(input_size=768,
                hidden_size=hidden_size,
                num_layers=num_layers, 
                num_classes=3)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_train_loss = 0.0
        correct_train_predictions = 0

        for batch in train_dataloader:
            embeddings = preprocessing.get_bert_embeddings(batch['input_ids'], batch['mask'])
            labels = batch['labels']

            optimizer.zero_grad()
            outputs = model(embeddings)            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_train_predictions += torch.sum(preds == labels)

        epoch_train_acc = correct_train_predictions.double() / len(train_dataset)
        epoch_train_loss = running_train_loss / len(train_dataloader)

        train_accuracies.append(epoch_train_acc.item())
        train_losses.append(epoch_train_loss)

        model.eval()
        running_test_loss = 0.0
        correct_test_predictions = 0

        with torch.no_grad():
            for batch in test_dataloader:
                embeddings = preprocessing.get_bert_embeddings(batch['input_ids'], batch['mask'])
                labels = batch['labels'].to(device)

                outputs = model(embeddings)

                loss = criterion(outputs, labels)
                running_test_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                correct_test_predictions += torch.sum(preds == labels)

            epoch_test_acc = correct_test_predictions.double() / len(test_dataset)
            epoch_test_loss = running_test_loss / len(test_dataloader)

            test_accuracies.append(epoch_test_acc.item())
            test_losses.append(epoch_test_loss)

    if epoch_test_loss < best_loss:
        best_loss = epoch_test_loss
        best_params = params
    
    print(f'run {run} '
          f'Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_acc:.4f}, '
          f'Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {epoch_test_acc:.4f}')
    
    print(f'current params: {params}')

    print(f"Current best loss: {best_loss:.4f} with params: {best_params}")