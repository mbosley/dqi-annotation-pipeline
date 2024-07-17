#!/usr/bin/env python3
import pandas as pd
import argparse
import os
import logging
from typing import List, Dict, Any
import yaml

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_file: str = 'config.yaml') -> Dict[str, Any]:
    try:
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        raise

def get_model_names(config: Dict[str, Any]) -> List[str]:
    return [model['name'] for model in config['models']]

def load_and_concatenate_results(config: Dict[str, Any], model_names: List[str]) -> pd.DataFrame:
    results_dfs = []
    for model_name in model_names:
        model_config = next((model for model in config['models'] if model['name'] == model_name), None)
        if not model_config:
            logger.warning(f"Model {model_name} not found in config")
            continue

        n_random_init = model_config['prompt'].get('n_random_init', 1)

        for init in range(n_random_init):
            file_path = os.path.join(config['data']['results_directory'], f"processed_results_{model_name}_init-{init}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['initialization'] = init
                df['model_name'] = model_name
                # Trim the '_*' from the unique_id
                df['unique_id'] = df['unique_id'].str.replace(r'_\d*$', '', regex=True)
                results_dfs.append(df)
            else:
                logger.warning(f"Results file for model {model_name}, initialization {init} not found: {file_path}")

    if not results_dfs:
        raise ValueError("No result files found.")

    return pd.concat(results_dfs, ignore_index=True)

def load_true_labels(config: Dict[str, Any]) -> pd.DataFrame:
    true_labels_path = config['data']['validation_data']
    return pd.read_csv(true_labels_path)

def map_column_names(df: pd.DataFrame, is_results: bool = True) -> Dict[str, List[str]]:
    column_mapping = {
        'Participation': ['participation', 'part'],
        'Justification-level': ['justification_level', 'just_level', 'justification-level'],
        'Justification-content': ['justification_content', 'just_content', 'justification-content'],
        'Respect-group': ['respect_groups', 'resp_group', 'respect-group'],
        'Respect-demand': ['respect_demand', 'resp_demand', 'respect-demand'],
        'Respect-counterarg': ['respect_counterargument', 'resp_counter', 'respect-counterarg'],
        'Constructive Politics': ['constructive_politics', 'constr_pol', 'constructive politics']
    }

    mapped_columns = {k: [] for k in column_mapping.keys()}
    for expected, possible_names in column_mapping.items():
        for col in df.columns:
            normalized_col = col.lower().replace('_', '-').replace(' ', '-')
            if any(name.lower().replace('_', '-') in normalized_col for name in possible_names):
                if is_results:
                    if 'score' in col.lower() or 'pred' in col.lower():
                        mapped_columns[expected].append(col)
                else:
                    mapped_columns[expected].append(col)

    logger.debug(f"Mapped columns ({'results' if is_results else 'true labels'}): {mapped_columns}")
    return mapped_columns

def merge_results_with_true_labels(results_df: pd.DataFrame, true_labels_df: pd.DataFrame) -> pd.DataFrame:
    if 'unique_id' not in results_df.columns or 'unique_id' not in true_labels_df.columns:
        raise ValueError("Both dataframes must have a 'unique_id' column for merging.")

    logger.debug(f"Results columns: {results_df.columns.tolist()}")
    logger.debug(f"True labels columns: {true_labels_df.columns.tolist()}")

    results_column_mapping = map_column_names(results_df, is_results=True)
    true_labels_column_mapping = map_column_names(true_labels_df, is_results=False)

    merged_df = pd.merge(results_df, true_labels_df, on='unique_id', how='left', suffixes=('_pred', '_true'))

    def extract_value(x):
        if isinstance(x, list) and len(x) == 1:
            return x[0]
        return x

    for category, results_cols in results_column_mapping.items():
        true_cols = true_labels_column_mapping.get(category, [])

        if results_cols and true_cols:
            merged_df[f'{category}_pred'] = merged_df[results_cols[0]].apply(extract_value)
            merged_df[f'{category}_true'] = merged_df[true_cols[0]].apply(extract_value)
        else:
            logger.warning(f"Could not map columns for {category}. Results cols: {results_cols}, True cols: {true_cols}")

    unmerged_count = merged_df['Speech#'].isnull().sum() if 'Speech#' in merged_df.columns else 0
    if unmerged_count > 0:
        logger.warning(f"{unmerged_count} rows from results_df did not merge with true labels.")

    logger.debug(f"Merged dataframe columns: {merged_df.columns.tolist()}")
    return merged_df

def prepare_data_for_evaluation(merged_df: pd.DataFrame) -> pd.DataFrame:
    categories = ['Participation', 'Justification-level', 'Justification-content',
                  'Respect-group', 'Respect-demand', 'Respect-counterarg', 'Constructive Politics']

    eval_columns = ['unique_id', 'model_name', 'initialization']

    # Add cost columns if they exist
    cost_columns = [col for col in merged_df.columns if 'cost' in col.lower()]
    eval_columns.extend(cost_columns)

    for category in categories:
        if f'{category}_pred' in merged_df.columns and f'{category}_true' in merged_df.columns:
            eval_columns.extend([f'{category}_pred', f'{category}_true'])
        else:
            logger.warning(f"Columns for {category} not found in the merged dataset")

    eval_df = merged_df[eval_columns]
    logger.info(f"Prepared {len(eval_df)} rows for evaluation out of {len(merged_df)} total rows")
    logger.debug(f"Evaluation dataframe columns: {eval_df.columns.tolist()}")

    return eval_df

def main():
    parser = argparse.ArgumentParser(description="Merge and validate LLM annotation results")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    model_names = get_model_names(config)

    logger.info(f"Processing models: {', '.join(model_names)}")

    logger.info("Loading and concatenating results...")
    results_df = load_and_concatenate_results(config, model_names)
    logger.info(f"Concatenated results shape: {results_df.shape}")

    logger.info("Loading true labels...")
    true_labels_df = load_true_labels(config)
    logger.info(f"True labels shape: {true_labels_df.shape}")

    # Ensure unique_id is string type in both dataframes
    results_df['unique_id'] = results_df['unique_id'].astype(str)
    true_labels_df['unique_id'] = true_labels_df['unique_id'].astype(str)

    logger.info("Merging results with true labels...")
    master_df = merge_results_with_true_labels(results_df, true_labels_df)
    logger.info(f"Master validation results shape: {master_df.shape}")

    logger.info("Preparing data for evaluation...")
    eval_df = prepare_data_for_evaluation(master_df)
    logger.info(f"Evaluation data shape: {eval_df.shape}")

    # Save the complete merged results as well
    complete_output_path = os.path.join(config['data']['results_directory'], "merged_results_all.csv")
    logger.info(f"Saving complete merged results to {complete_output_path}")
    eval_df.to_csv(complete_output_path, index=False)
    logger.info("All results saved successfully.")

    # Print some basic statistics
    logger.info("\nBasic Statistics:")
    logger.info(f"Total number of annotations: {len(eval_df)}")
    logger.info(f"Number of unique speeches: {eval_df['unique_id'].nunique()}")
    logger.info(f"Models included: {eval_df['model_name'].unique()}")
    logger.info(f"Initializations per model: {eval_df.groupby('model_name')['initialization'].nunique().to_dict()}")
    logger.info(f"Columns in final dataset: {eval_df.columns.tolist()}")

if __name__ == "__main__":
    main()
