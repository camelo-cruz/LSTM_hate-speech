import pandas as pd
import nltk
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')  # Download WordNet for lemmatization
nltk.download('omw-1.4')  # Download additional WordNet resources

# Set up stopwords in English and initialize lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_data(csv_file):
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

    # Tokenize the text column, remove stopwords, and lemmatize
    text_df['comment'] = text_df['comment'].apply(
        lambda x: [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(x)
                   if word.lower() not in stop_words and word.isalpha() and word != '']  # Exclude digits
    )

    return text_df, labels_df


def build_vocab(tokenized_comments):
    vocab = defaultdict(lambda: len(vocab))  # Automatically assigns indices to new words
    vocab["<PAD>"] = 0  # Assign special index 0 for padding
    for comment in tokenized_comments:
        for word in comment:
            vocab[word]  # Add word to vocab
    return vocab

def convert_to_indices(tokenized_comments, vocab):
    return [[vocab[word] for word in comment] for comment in tokenized_comments]

def pad_sequences(sequences, max_len=100):
    padded_sequences = torch.zeros((len(sequences), max_len), dtype=torch.long)  # Create tensor of zeros for padding
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_len)
        padded_sequences[i, :seq_len] = torch.tensor(seq[:seq_len], dtype=torch.long)
    return padded_sequences