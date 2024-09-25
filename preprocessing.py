import pandas as pd
import torch
from transformers import BertTokenizer
import nltk
from nltk.corpus import stopwords
import re
import emoji
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

# Preprocessing function using BERT tokenizer and train-test split
def preprocess_data(csv_file, max_len=100, test_size=0.3):
    df = pd.read_csv(csv_file)

    text_df = df[['comment']].copy()

    label_encoder = LabelEncoder()
    labels_df = label_encoder.fit_transform(df['label'])

    text_df = text_df.reset_index(drop=True)
    labels_df = pd.Series(labels_df).reset_index(drop=True)

    text_df['comment'] = text_df['comment'].apply(clean_text)
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        text_df['comment'], labels_df,
        test_size=0.33, random_state=42, 
    )

    # Tokenize, pad, and create attention masks for training and testing sets
    train_encoding = tokenizer(
        train_texts.tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt',
        return_attention_mask=True
    )

    test_encoding = tokenizer(
        test_texts.tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt',
        return_attention_mask=True
    )

    train_input_ids = train_encoding['input_ids'].to(device)
    train_attention_mask = train_encoding['attention_mask'].to(device)
    train_labels = torch.tensor(train_labels.values, dtype=torch.long).to(device)

    test_input_ids = test_encoding['input_ids'].to(device)
    test_attention_mask = test_encoding['attention_mask'].to(device)
    test_labels = torch.tensor(test_labels.values, dtype=torch.long).to(device)


    train_sample = {
        'input_ids': train_input_ids, 
        'attention_mask': train_attention_mask, 
        'labels': train_labels,
    }

    test_sample = {
        'input_ids': test_input_ids, 
        'attention_mask': test_attention_mask, 
        'labels': test_labels,
    }

    return train_sample, test_sample


def decode_input_ids(input_ids, skip_special_tokens=True):
    # Use the tokenizer's decode method to convert token IDs back to text
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=skip_special_tokens)
    return decoded_text

def get_vocab_size():
    return tokenizer.vocab_size