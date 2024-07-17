#!/usr/bin/env python3
import pandas as pd
import numpy as np
import yaml
import argparse
from typing import Dict, Any
import gc
import math
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_file: str = 'config.yaml') -> Dict[str, Any]:
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def load_data(train_data_path: str, test_data_path: str, merged_data_path: str) -> tuple:
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    return train_data, test_data, merged_data_path

def filter_merged_data(merged_data_path: str, speech_ids: set) -> pd.DataFrame:
    chunk_size = 100000
    filtered_chunks = []

    for chunk in pd.read_csv(merged_data_path, chunksize=chunk_size):
        filtered_chunk = chunk[chunk['speech_id'].isin(speech_ids)]
        if not filtered_chunk.empty:
            filtered_chunks.append(filtered_chunk)

    return pd.concat(filtered_chunks, ignore_index=True)

def get_speech_text(speech_ids: list, merged_data: pd.DataFrame) -> str:
    speeches = []
    for speech_id in speech_ids:
        if pd.notna(speech_id):
            speech_row = merged_data[merged_data['speech_id'] == float(speech_id)]
            if not speech_row.empty:
                speaker = speech_row['speaker'].iloc[0]
                speech = speech_row['speech'].iloc[0]
                speeches.append(f"{speaker}: {speech}")
    return " ".join(speeches)

def get_previous_speeches(speech_number: int, data: pd.DataFrame, merged_data: pd.DataFrame, k: int, n: int, m: int) -> str:
    previous_speeches = []
    for i in range(max(0, int(speech_number)-k), int(speech_number)):
        speech_ids = data.loc[data['Speech#'] == i, ['speech_id_1', 'speech_id_2', 'speech_id_3', 'speech_id_4']].values.flatten()
        speech_text = get_speech_text(speech_ids, merged_data)
        previous_speeches.append(speech_text[:m])
    previous_speeches_text = " ".join(previous_speeches)
    return previous_speeches_text[:n]

import math

def generate_annotation_rationales(row: pd.Series) -> Dict[str, str]:
    rationales = {}
    unable_to_determine_count = 0

    def safe_int(value):
        try:
            if math.isnan(value):
                return None
            return int(value)
        except:
            return None

    # 1. Participation
    participation_score = safe_int(row['Participation'])
    if participation_score == 0:
        rationales["Participation"] = f"Score: 0 - Participation was impaired - speaker was cut off or explicitly disturbed."
    elif participation_score == 1:
        rationales["Participation"] = f"Score: 1 - Normal participation was possible."
    else:
        rationales["Participation"] = f"Score: {participation_score} - Unable to determine participation level."
        unable_to_determine_count += 1

    # 2.1 Level of Justification
    justification_level_score = safe_int(row['Justification-level'])
    if justification_level_score == 0:
        rationales["Justification-level"] = f"Score: 0 - No justification provided."
    elif justification_level_score == 1:
        rationales["Justification-level"] = f"Score: 1 - Inferior justification - reason given but not properly linked to demand."
    elif justification_level_score == 2:
        rationales["Justification-level"] = f"Score: 2 - Qualified justification - one complete linkage between demand and reason."
    elif justification_level_score == 3:
        rationales["Justification-level"] = f"Score: 3 - Sophisticated justification (broad) - multiple complete justifications."
    elif justification_level_score == 4:
        rationales["Justification-level"] = f"Score: 4 - Sophisticated justification (in depth) - multiple justifications with one embedded in complete inference chain."
    else:
        rationales["Justification-level"] = f"Score: {justification_level_score} - Unable to determine justification level."
        unable_to_determine_count += 1

    # 2.2 Content of Justification
    justification_content_score = safe_int(row['Justification-content'])
    if justification_content_score == 0:
        rationales["Justification-content"] = f"Score: 0 - Explicit reference to group/constituency interests."
    elif justification_content_score == 1:
        rationales["Justification-content"] = f"Score: 1 - Neutral statement, no explicit references to group interests."
    elif justification_content_score == 2:
        rationales["Justification-content"] = f"Score: 2 - Explicit reference to common good (utilitarian/collective)."
    elif justification_content_score == 3:
        rationales["Justification-content"] = f"Score: 3 - Explicit reference to helping least advantaged (difference principle)."
    else:
        rationales["Justification-content"] = f"Score: {justification_content_score} - Unable to determine justification content."
        unable_to_determine_count += 1

    # 3.1 Respect toward Groups
    respect_group_score = safe_int(row['Respect-group'])
    if respect_group_score == 0:
        rationales["Respect-group"] = f"Score: 0 - Explicitly negative statement about group to be helped."
    elif respect_group_score == 1:
        rationales["Respect-group"] = f"Score: 1 - Neither negative nor positive statement about group."
    elif respect_group_score == 2:
        rationales["Respect-group"] = f"Score: 2 - Explicitly positive statement about group."
    else:
        rationales["Respect-group"] = f"Score: {respect_group_score} - Unable to determine respect toward groups."
        unable_to_determine_count += 1

    # 3.2 Respect toward Demands
    respect_demand_score = safe_int(row['Respect-demand'])
    if respect_demand_score == 0:
        rationales["Respect-demand"] = f"Score: 0 - Explicitly negative statement about demand."
    elif respect_demand_score == 1:
        rationales["Respect-demand"] = f"Score: 1 - Neither negative nor positive statement about demand."
    elif respect_demand_score == 2:
        rationales["Respect-demand"] = f"Score: 2 - Explicitly positive statement about demand."
    elif respect_demand_score == 3:
        rationales["Respect-demand"] = f"Score: 3 - Agreeing with and positively valuing demand."
    else:
        rationales["Respect-demand"] = f"Score: {respect_demand_score} - Unable to determine respect toward demands."
        unable_to_determine_count += 1

    # 3.3 Respect toward Counterarguments
    respect_counterarg_score = safe_int(row['Respect-counterarg'])
    if respect_counterarg_score == 0:
        rationales["Respect-counterarg"] = f"Score: 0 - Counterarguments degraded."
    elif respect_counterarg_score == 1:
        rationales["Respect-counterarg"] = f"Score: 1 - Counterarguments ignored."
    elif respect_counterarg_score == 2:
        rationales["Respect-counterarg"] = f"Score: 2 - Counterarguments included but not valued."
    elif respect_counterarg_score == 3:
        rationales["Respect-counterarg"] = f"Score: 3 - Counterarguments explicitly valued."
    elif respect_counterarg_score == 4:
        rationales["Respect-counterarg"] = f"Score: 4 - Counterarguments agreed with."
    else:
        rationales["Respect-counterarg"] = f"Score: {respect_counterarg_score} - Unable to determine respect toward counterarguments."
        unable_to_determine_count += 1

    # 4. Constructive Politics
    constructive_politics_score = safe_int(row['Constructive Politics'])
    if constructive_politics_score == 0:
        rationales["Constructive Politics"] = f"Score: 0 - Positional politics - merely reiterating one's position."
    elif constructive_politics_score == 1:
        rationales["Constructive Politics"] = f"Score: 1 - Alternative proposal - mediating but off current agenda."
    elif constructive_politics_score == 2:
        rationales["Constructive Politics"] = f"Score: 2 - Consensus appeal - unspecific call for compromise."
    elif constructive_politics_score == 3:
        rationales["Constructive Politics"] = f"Score: 3 - Mediating proposal - specific compromise on current agenda."
    else:
        rationales["Constructive Politics"] = f"Score: {constructive_politics_score} - Unable to determine level of constructive politics."
        unable_to_determine_count += 1

    # if unable_to_determine_count > 0:
    #     logger.warning(f"Unable to determine {unable_to_determine_count} categories for speech {row['unique_id']}")

    return rationales

def process_data(data: pd.DataFrame, merged_data: pd.DataFrame, is_train: bool, model_config: Dict[str, Any]) -> pd.DataFrame:
    prompt_config = model_config['prompt']
    k, n, m, p = prompt_config.get('k', 4), prompt_config.get('n', 5000), prompt_config.get('m', 1000), prompt_config.get('p', 10000)

    result = []
    total_speeches = len(data)
    speeches_with_issues = 0

    for _, row in data.iterrows():
        speech_ids = row[['speech_id_1', 'speech_id_2', 'speech_id_3', 'speech_id_4']].dropna().astype(float).values
        speeches = get_speech_text(speech_ids, merged_data)[:p]
        previous_speeches = get_previous_speeches(row['Speech#'], data, merged_data, k, n, m) if prompt_config.get('include_previous_speeches', False) else ""

        processed_row = {
            'Speech#': row['Speech#'],
            'speeches': speeches,
            'previous_speeches': previous_speeches,
            'unique_id': row['unique_id'],
            'Participation': row['Participation'],
            'Justification-level': row['Justification-level'],
            'Justification-content': row['Justification-content'],
            'Respect-group': row['Respect-group'],
            'Respect-demand': row['Respect-demand'],
            'Respect-counterarg': row['Respect-counterarg'],
            'Constructive Politics': row['Constructive Politics']
        }

        if is_train:
            rationales = generate_annotation_rationales(row)
            processed_row['annotation_rationales'] = json.dumps(rationales)
            if any("Unable to determine" in rationale for rationale in rationales.values()):
                speeches_with_issues += 1
       

        result.append(processed_row)

    logger.info(f"Processed {total_speeches} speeches. {speeches_with_issues} had at least one undetermined category.")
    return pd.DataFrame(result)

def main():
    parser = argparse.ArgumentParser(description="Process data for a specific model")
    parser.add_argument("--model", required=True, help="Model name to process data for")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    print(f"Processing data for model: {args.model}")

    config = load_config(args.config)
    model_config = next((model for model in config['models'] if model['name'] == args.model), None)
    if not model_config:
        raise ValueError(f"Model {args.model} not found in config")

    train_data_path = config['data']['train_data_file']
    test_data_path = config['data']['test_data_file']
    merged_data_path = config['data']['merged_data_file']

    train_data, test_data, merged_data_path = load_data(train_data_path, test_data_path, merged_data_path)

    all_speech_ids = set()
    for df in [train_data, test_data]:
        for col in ['speech_id_1', 'speech_id_2', 'speech_id_3', 'speech_id_4']:
            all_speech_ids.update(df[col].dropna().astype(float).unique())

    merged_data = filter_merged_data(merged_data_path, all_speech_ids)

    processed_train_data = process_data(train_data, merged_data, is_train=True, model_config=model_config)
    processed_test_data = process_data(test_data, merged_data, is_train=False, model_config=model_config)

    train_output_path = f"{config['data']['clean_data_directory']}/processed_train_data_{args.model}.csv"
    test_output_path = f"{config['data']['clean_data_directory']}/processed_test_data_{args.model}.csv"

    processed_train_data.to_csv(train_output_path, index=False)
    processed_test_data.to_csv(test_output_path, index=False)

    print("Processed data saved successfully.")

if __name__ == "__main__":
    main()
