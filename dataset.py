import pandas as pd
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer, BertModel

class TextDataset(Dataset):
    def __init__(self, csv_file, bert_model_name='bert-base-uncased', max_len=128):
        """
        Custom dataset for loading text, labels, one-hot encoded contextual features,
        and generating BERT embeddings.

        Parameters:
        ----------
        csv_file : str
            Path to the CSV file containing the dataset.
        bert_model_name : str, optional
            The name of the pre-trained BERT model to use.
        max_len : int, optional
            Maximum length of tokens.
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        comment = self.data.iloc[idx]["comment"]
        label = self.data.iloc[idx]["label"]

        # Tokenize the text for BERT input
        tokens = self.tokenizer(
            comment,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Pass the tokens through the BERT model to get the embeddings
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = self.bert_model(
                input_ids=tokens['input_ids'].squeeze().unsqueeze(0),  # BERT expects batched input
                attention_mask=tokens['attention_mask'].squeeze().unsqueeze(0)
            )
        
        # Get the last hidden state (the embeddings for each token)
        bert_embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: [max_len, hidden_size]

        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'label': label,
            'context': context,  # Return one-hot encoded context features
            'bert_embeddings': bert_embeddings  # Return BERT embeddings
        }
