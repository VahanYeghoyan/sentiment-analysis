import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    return df


def split_data(df, test_size=0.2, random_state=42):
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, val_df