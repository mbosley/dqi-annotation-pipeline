#!/usr/bin/env python3
import pandas as pd
import os
import argparse
import logging
import yaml
import concurrent.futures

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_file: str = 'config.yaml') -> dict:
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def process_file_id(file_id, directory_path):
    logging.info(f"Processing files for {file_id}...")

    speaker_map_file = os.path.join(directory_path, f"{file_id}_SpeakerMap.txt")
    descr_file = os.path.join(directory_path, f"descr_{file_id}.txt")
    speeches_file = os.path.join(directory_path, f"speeches_{file_id}.txt")

    if os.path.exists(speaker_map_file) and os.path.exists(descr_file) and os.path.exists(speeches_file):
        try:
            speaker_map = pd.read_csv(speaker_map_file, sep='|', dtype=str, encoding='latin1')
            descr = pd.read_csv(descr_file, sep='|', dtype=str, encoding='latin1')
            speeches = pd.read_csv(speeches_file, sep='|', dtype=str, encoding='latin1', on_bad_lines='skip')
        except UnicodeDecodeError:
            logging.warning(f"Encoding error in files for {file_id}, attempting with 'ISO-8859-1' encoding.")
            speaker_map = pd.read_csv(speaker_map_file, sep='|', dtype=str, encoding='ISO-8859-1')
            descr = pd.read_csv(descr_file, sep='|', dtype=str, encoding='ISO-8859-1')
            speeches = pd.read_csv(speeches_file, sep='|', dtype=str, encoding='ISO-8859-1', on_bad_lines='skip')

        descr.rename(columns={'last_name': 'lastname'}, inplace=True)
        merged_data = pd.merge(speeches, descr, on=['speech_id'], how='left')
        final_merged_data = pd.merge(merged_data, speaker_map, on=['speech_id', 'lastname', 'chamber', 'gender', 'state'], how='left')
        return final_merged_data
    else:
        logging.warning(f"Missing files for {file_id}, skipping this batch.")
        return None

def merge_datasets(directory_path, output_file_path):
    all_merged_data = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file_id, f"{i:03}", directory_path) for i in range(97, 109)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                all_merged_data.append(result)

    concatenated_data = pd.concat(all_merged_data, ignore_index=True)
    concatenated_data.to_csv(output_file_path, index=False)

    concatenated_data['date'] = pd.to_datetime(concatenated_data['date'], format='%Y%m%d').dt.strftime('%m/%d/%Y')
    concatenated_data['legislature'] = concatenated_data['chamber'].apply(lambda x: 'Senate' if x == 'S' else 'House')
    concatenated_data.rename(columns={'lastname': 'last_name'}, inplace=True)

    return concatenated_data

def main():
    parser = argparse.ArgumentParser(description="Clean label data and perform train-test split")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    config = load_config(args.config)

    input_directory = config['data']['raw_debates_dir']
    output_file_path = config['data']['merged_data_file']

    merged_data = merge_datasets(input_directory, output_file_path)
    subset_data = merged_data[['chamber', 'date', 'speech_id', 'speaker', 'speech']]
    subset_data.to_csv(os.path.join(os.path.dirname(output_file_path), 'merged_us_data_subset.csv'), index=False)

if __name__ == "__main__":
    main()
