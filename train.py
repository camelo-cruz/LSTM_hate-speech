import os
import torch
import random
import preprocessing
import numpy as np
import torch.nn as nn
import torch.optim as optim
from plot import plot_train_test
from torch.utils.data import DataLoader
from dataset import TextDataset
from model import LSTM

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_dir = os.getcwd()
csv_path = os.path.join(current_dir, 'data', 'final_hateXplain.csv')

#define hyperparameters
input_len = 128
hidden_size = 128
num_layers = 3
num_classes = 3
batch_size = 32
num_epochs = 100
learning_rate = 0.001
embedding_dim = 128

train_sample, test_sample = preprocessing.preprocess_data(csv_path, input_len)
train_dataset = TextDataset(train_sample['input_ids'], train_sample['attention_mask'], train_sample['labels'])
test_dataset = TextDataset(test_sample['input_ids'], test_sample['attention_mask'], test_sample['labels'])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


vocab_size = preprocessing.get_vocab_size()
model = LSTM(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=num_classes
)

model.to(device)

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# Loss function and optimizer
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

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

        # Compute accuracy
        _, preds = torch.max(outputs, dim=1)
        correct_train_predictions += torch.sum(preds == labels)

    # Compute and store training loss and accuracy
    epoch_train_loss = running_train_loss / len(train_dataloader)
    epoch_train_acc = correct_train_predictions.double() / len(train_dataset)
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc.item())

    # Validation/Test
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
          f'Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_acc:.4f}, '
          f'Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {epoch_test_acc:.4f}')

plot_train_test(train_accuracies, test_accuracies, train_losses, test_losses, 'train metrics')
