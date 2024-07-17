#!/usr/bin/env python3
import pandas as pd
import hashlib
from sklearn.model_selection import train_test_split
import yaml
import argparse

def load_config(config_file: str = 'config.yaml') -> dict:
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def generate_unique_id(row):
    unique_string = f"{row['Speech#']}_{row['date']}_{row['topic']}_{row['legislature']}_{row['last_name']}_{row['party']}_{row['gender']}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def clean_dataset(file_path):
    data = pd.read_csv(file_path)

    data['debate_metadata'] = data['Speaker'].where(data['Speech#'].isna())
    data['debate_metadata'].ffill(inplace=True)
    data = data.dropna(subset=['Speech#'])

    data[['debate_name', 'date']] = data['debate_metadata'].str.rsplit(' ', n=1, expand=True)
    data['topic'] = data['debate_name'].apply(lambda x: ' '.join(x.strip().rsplit(' ', 1)[:-1]))
    data['legislature'] = data['debate_name'].apply(lambda x: x.strip().rsplit(' ', 1)[-1])
    data[['last_name', 'party', 'gender']] = data['Speaker'].str.split('/', expand=True)

    data['unique_id'] = data.apply(generate_unique_id, axis=1)
    data.drop(['Speaker', 'debate_metadata', 'debate_name'], axis=1, inplace=True)

    columns = ['unique_id'] + [col for col in data.columns if col != 'unique_id']
    return data[columns]

def main():
    parser = argparse.ArgumentParser(description="Clean label data and perform train-test split")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    file_path = config['data']['input_file']
    test_size = config['data']['test_size']
    output_dir = config['data']['clean_data_directory']

    cleaned_data = clean_dataset(file_path)

    # Perform train-test split
    train_data, test_data = train_test_split(cleaned_data, test_size=test_size, random_state=42)

    # Save the cleaned and split datasets
    train_data.to_csv(f"{output_dir}/train_data.csv", index=False)
    test_data.to_csv(f"{output_dir}/test_data.csv", index=False)

    print(f"Cleaned and split data saved to {output_dir}")
    print(f"Train set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")

if __name__ == "__main__":
    main()
