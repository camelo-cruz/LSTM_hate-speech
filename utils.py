import pandas as pd
import torch
from transformers import BertTokenizer
import nltk
from nltk.corpus import stopwords
import re
import emoji


# Detect if CUDA is available, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download required stopwords
nltk.download('stopwords')

# Initialize the BERT tokenizer from Hugging Face
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load NLTK stopwords
stop_words = set(stopwords.words('english'))

# Function to clean text by removing stopwords and numbers
def clean_text(text):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Remove stopwords and words that contain digits
    cleaned_words = [emoji.demojize(word) for word in words if word.lower() not in stop_words and not re.search(r'\d', word)]

    # Join the cleaned words back into a single string
    return ' '.join(cleaned_words)

# Preprocessing function using BERT tokenizer
def preprocess_data(csv_file, max_len=100):
    df = pd.read_csv(csv_file)

    # Separate the text column
    text_df = df[['comment']].copy()

    # Map labels to class indices
    label_mapping = {
        'normal': 0,
        'offensive': 1,
        'hatespeech': 2
    }
    labels_df = df['label'].map(label_mapping)

    # Reset index for all DataFrames
    text_df = text_df.reset_index(drop=True)
    labels_df = labels_df.reset_index(drop=True)

    # Clean the text by removing stopwords and numbers
    text_df['comment'] = text_df['comment'].apply(clean_text)

    # Use BERT tokenizer to tokenize, pad, and create attention masks
    encoding = tokenizer(
        text_df['comment'].tolist(),  # The list of cleaned comments
        truncation=True,
        padding='max_length',  # Pad to max_len
        max_length=max_len,  # Max length of input tokens
        return_tensors='pt',  # Return PyTorch tensors
        return_attention_mask=True  # Return attention mask (for padding)
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    labels = torch.tensor(labels_df.values, dtype=torch.long).to(device)

    return input_ids, attention_mask, labels

def decode_input_ids(input_ids, skip_special_tokens=True):
    # Use the tokenizer's decode method to convert token IDs back to text
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=skip_special_tokens)
    return decoded_text

def get_vocab_size():
    return tokenizer.vocab_size