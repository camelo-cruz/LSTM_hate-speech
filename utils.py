import pandas as pd

def preprocess_data(csv_file, output_file):
    """
    Preprocess the dataset by one-hot encoding the context columns and saving the processed dataset.

    Parameters:
    ----------
    csv_file : str
        Path to the original CSV file containing the dataset.
    output_file : str
        Path to the output CSV file where the processed data will be saved.
    """
    df = pd.read_csv(csv_file)
    
    df_encoded = pd.get_dummies(df, columns=['Race', 'Religion', 'Gender', 'Sexual Orientation', 'Miscellaneous'])
    
    return df_encoded