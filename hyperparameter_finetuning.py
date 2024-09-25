import os
import torch
import preprocessing
import random
import numpy as np
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

# Define hyperparameter ranges
learning_rates = [1e-3, 1e-4, 1e-5]
batch_sizes = [32, 64]
hidden_sizes = [128, 256]
num_layers_list = [2, 3]
input_len = [32]
num_epochs = 100

# Define the combinations
param_combinations = list(product(learning_rates, batch_sizes, hidden_sizes, num_layers_list, input_len))

best_accuracy = 0
best_params = None

vocab_size = preprocessing.get_vocab_size()

for params in param_combinations:
    learning_rate, batch_size, hidden_size, num_layers, input_len = params

    train_sample, test_sample = preprocessing.preprocess_data(csv_path, input_len)
    train_dataset = TextDataset(train_sample['input_ids'], train_sample['attention_mask'], train_sample['labels'])
    test_dataset = TextDataset(test_sample['input_ids'], test_sample['attention_mask'], test_sample['labels'])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    model = LSTM(vocab_size=vocab_size, 
                embedding_dim=128, 
                hidden_size=hidden_size,
                num_layers=num_layers, 
                num_classes=3)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        correct_train_predictions = 0

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_train_predictions += torch.sum(preds == labels)

        accuracy_train = correct_train_predictions.double() / len(train_dataset)
        loss_train = running_train_loss / len(train_dataloader)

        train_losses.append(loss_train)
        train_accuracies.append(accuracy_train.item())

        model.eval()
        running_test_loss = 0.0
        correct_test_predictions = 0

        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids)

                loss = criterion(outputs, labels)
                running_test_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                correct_test_predictions += torch.sum(preds == labels)

        epoch_test_loss = running_test_loss / len(test_dataloader)
        epoch_test_acc = correct_test_predictions.double() / len(test_dataset)

        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_acc.item())

        print(f'Epoch {epoch+1}/{num_epochs}, '
            f'Train Loss: {loss_train:.4f}, Train Accuracy: {accuracy_train:.4f}, '
            f'Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {epoch_test_acc:.4f}')

        if epoch_test_acc > best_accuracy:
            best_accuracy = epoch_test_acc
            best_params = params

    print(f"Current best accuracy: {best_accuracy:.4f} with params: {best_params}")