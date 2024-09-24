import os
import torch
from .utils import preprocessing
from itertools import product
from model import LSTM
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TextDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_dir = os.getcwd()
csv_path = os.path.join(current_dir, 'data', 'final_hateXplain.csv')

train_sample, test_sample = preprocessing.preprocess_data(csv_path, input_len)
train_dataset = TextDataset(train_sample['input_ids'], train_sample['attention_mask'], train_sample['labels'])
test_dataset = TextDataset(test_sample['input_ids'], test_sample['attention_mask'], test_sample['labels'])

# Define hyperparameter ranges
learning_rates = [1e-3, 1e-4, 1e-5]
batch_sizes = [32, 64]
hidden_sizes = [128, 256]
num_layers_list = [2, 3]
num_epochs = 100

# Define the combinations
param_combinations = list(product(learning_rates, batch_sizes, hidden_sizes, num_layers_list))

best_accuracy = 0
best_params = None

vocab_size = preprocessing.get_vocab_size()

for params in param_combinations:
    learning_rate, batch_size, hidden_size, num_layers = params
    
    # Rebuild the model with current hyperparameters
    model = LSTM(vocab_size=vocab_size, embedding_dim=128, hidden_size=hidden_size, num_layers=num_layers, num_classes=3)
    model.to(device)
    
    # Define the optimizer with the current learning rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Define the DataLoader with the current batch size
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    
    # Train the model (simplified version)
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)

        accuracy = correct_predictions.double() / len(train_dataset)

        # Check if current model is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    print(f"Current best accuracy: {best_accuracy:.4f} with params: {best_params}")
