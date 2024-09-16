import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TextDataset
import utils

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
current_dir = os.getcwd()
csv_path = os.path.join(current_dir, 'data', 'final_hateXplain.csv')

# Preprocess data
text_df, labels_df = utils.preprocess_data(csv_path)
vocab = utils.build_vocab(text_df['comment'])
indices = utils.convert_to_indices(text_df['comment'], vocab)
padded = utils.pad_sequences(indices)
print(padded)

# Define hyperparameters
input_len = 128
hidden_size = 128
num_layers = 4
num_classes = 3
batch_size = 32
num_epochs = 100
learning_rate = 0.001

# Instantiate the dataset and dataloader
dataset = TextDataset(
    text=text_df,
    labels=labels_df,
    bert_model_name='bert-base-uncased',
    max_len=input_len
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        output, (hidden, cell) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

# Instantiate the model
vocab_size = dataset.tokenizer.vocab_size
embedding_dim = 128  # Can be adjusted
model = LSTM(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=num_classes
)

model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Compute accuracy
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct_predictions.double() / len(dataset)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
