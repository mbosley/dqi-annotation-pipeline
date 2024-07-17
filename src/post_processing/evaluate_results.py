#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from scipy.stats import spearmanr, ttest_ind, pearsonr
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import LabelBinarizer
import argparse
import logging
from logging.handlers import RotatingFileHandler
from tqdm import tqdm
import yaml
import json
from typing import Dict, List, Any, Tuple
import os
from datetime import datetime
import warnings

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        raise

def setup_logging(config: Dict[str, Any], script_name: str) -> None:
    """Set up logging with per-script log files and rotation."""
    log_config = config['logging']
    log_dir = log_config['directory']
    log_level = logging.getLevelName(config['global']['log_level'])
    log_format = log_config['format']
    date_format = log_config['date_format']

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"{script_name}.log")

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=log_config['max_file_size'],
        backupCount=log_config['backup_count']
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    logging.root.setLevel(log_level)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)

    logging.info(f"Logging initialized for {script_name}")

def load_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load data from the merged results CSV file."""
    file_path = os.path.join(config['data']['results_directory'], "merged_results_all.csv")
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

def get_model_names(config: Dict[str, Any]) -> List[str]:
    """Get the list of unique models from the configuration."""
    return [model['name'] for model in config['models']]

def ordinal_classification_index(y_true, y_pred):
    """Calculate the Ordinal Classification Index (OCI)."""
    n = len(y_true)
    num_classes = max(max(y_true), max(y_pred)) + 1
    max_dist = num_classes - 1

    sum_distances = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred))
    oci = 1 - (sum_distances / (n * max_dist))

    return oci

def calculate_metrics(df: pd.DataFrame, model_name: str, init: int, config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Calculate performance metrics and confusion matrices for a given model and initialization."""
    metrics = {}
    confusion_matrices = {}
    try:
        model_data = df[(df['model_name'] == model_name) & (df['initialization'] == init)]

        dimensions = [
            'Participation',
            'Justification-level',
            'Justification-content',
            'Respect-group',
            'Respect-demand',
            'Respect-counterarg',
            'Constructive Politics'
        ]

        for dimension in dimensions:
            true_column = f"{dimension}_true"
            pred_column = f"{dimension}_pred"

            if true_column not in model_data.columns or pred_column not in model_data.columns:
                logging.warning(f"Columns {true_column} or {pred_column} not found in the data for {model_name} (init {init})")
                continue

            # Filter out missing values and special cases (88 or 99)
            valid_data = model_data[(model_data[true_column] != 88) &
                                    (model_data[true_column] != 99) &
                                    (model_data[true_column].notna()) &
                                    (model_data[pred_column].notna())]

            if len(valid_data) > 0:
                true_values = valid_data[true_column]
                pred_values = valid_data[pred_column]

                metrics[dimension] = calculate_dimension_metrics(true_values, pred_values, config)

                # Calculate confusion matrix
                labels = sorted(set(true_values) | set(pred_values))
                cm = confusion_matrix(true_values, pred_values, labels=labels)
                confusion_matrices[dimension] = {
                    'matrix': cm.tolist(),
                    'labels': labels
                }
            else:
                logging.warning(f"No valid data for {dimension} in {model_name} (init {init})")

    except Exception as e:
        logging.error(f"Error calculating metrics for {model_name} (init {init}): {e}")
        raise

    return metrics, confusion_matrices

def calculate_dimension_metrics(true_values: pd.Series, pred_values: pd.Series, config: Dict[str, Any]) -> Dict[str, float]:
    """Calculate metrics for a single dimension."""
    dimension_metrics = {}
    for metric in config['analysis']['metrics_to_analyze']:
        if metric == 'mse':
            dimension_metrics[metric] = mean_squared_error(true_values, pred_values)
        elif metric == 'mae':
            dimension_metrics[metric] = mean_absolute_error(true_values, pred_values)
        elif metric == 'correlation':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr, _ = spearmanr(true_values, pred_values)
            dimension_metrics[metric] = corr if not np.isnan(corr) else 0
        elif metric == 'accuracy':
            dimension_metrics[metric] = accuracy_score(true_values, pred_values)
        elif metric == 'f1':
            dimension_metrics[metric] = f1_score(true_values, pred_values, average='weighted')
        elif metric == 'oci':
            dimension_metrics[metric] = ordinal_classification_index(true_values, pred_values)
    return dimension_metrics

def calculate_aggregate_metrics(metrics: Dict[str, Dict[str, float]], config: Dict[str, Any]) -> Dict[str, float]:
    """Calculate aggregate metrics across all dimensions."""
    try:
        aggregate = {}
        for metric in config['analysis']['aggregate_metrics']:
            if metric.startswith('avg_'):
                base_metric = metric[4:]
                aggregate[metric] = np.mean([m[base_metric] for m in metrics.values() if base_metric in m])
        return aggregate
    except Exception as e:
        logging.error(f"Error calculating aggregate metrics: {e}")
        raise

def calculate_cost_metrics(df: pd.DataFrame, model_name: str, init: int) -> Dict[str, float]:
    """Calculate cost metrics for a given model and initialization."""
    model_data = df[(df['model_name'] == model_name) & (df['initialization'] == init)]
    cost_columns = [col for col in model_data.columns if 'cost' in col.lower()]

    cost_metrics = {}
    for col in cost_columns:
        cost_metrics[f'total_{col}'] = model_data[col].sum()
        cost_metrics[f'average_{col}'] = model_data[col].mean()

    cost_metrics['total_cost'] = sum(cost_metrics[f'total_{col}'] for col in cost_columns)
    cost_metrics['average_cost'] = cost_metrics['total_cost'] / len(model_data)

    return cost_metrics

def perform_significance_tests(df: pd.DataFrame, models: List[str], config: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Perform significance tests between models and validation data."""
    significance_results = {"t_tests": {}, "tukey_hsd": {}}
    try:
        for column in df.columns:
            if column.endswith('_pred'):
                true_column = f"{column[:-5]}_true"
                dimension = true_column[:-5]
                significance_results["t_tests"][dimension] = {}
                significance_results["tukey_hsd"][dimension] = {}

                if config['analysis']['tests']['perform_t_tests']:
                    for model in models:
                        model_data = df[df['model_name'] == model][column]
                        true_data = df[df['model_name'] == model][true_column]
                        valid_data = pd.DataFrame({'pred': model_data, 'true': true_data}).dropna()
                        if len(valid_data) > 1 and not (valid_data['pred'].var() == 0 or valid_data['true'].var() == 0):
                            t_stat, p_value = ttest_ind(valid_data['pred'], valid_data['true'])
                            significance_results["t_tests"][dimension][f"{model}_vs_true"] = p_value
                        else:
                            significance_results["t_tests"][dimension][f"{model}_vs_true"] = None

                if config['analysis']['tests']['perform_tukey_hsd']:
                    model_data = [df[df['model_name'] == model][column].dropna() for model in models]
                    model_labels = np.concatenate([[model] * len(data) for model, data in zip(models, model_data)])
                    all_data = np.concatenate(model_data)

                    if len(all_data) > 1 and len(set(all_data)) > 1:
                        tukey_results = pairwise_tukeyhsd(all_data, model_labels)
                        for i, (group1, group2, p_value) in enumerate(zip(tukey_results.groupsunique, tukey_results.groupsunique[1:], tukey_results.pvalues)):
                            significance_results["tukey_hsd"][dimension][f"{group1}_vs_{group2}"] = p_value
                    else:
                        for i, model1 in enumerate(models):
                            for j, model2 in enumerate(models):
                                if i < j:
                                    significance_results["tukey_hsd"][dimension][f"{model1}_vs_{model2}"] = None
    except Exception as e:
        logging.error(f"Error performing significance tests: {e}")
        raise
    return significance_results

def safe_standardize(X):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    X_std = np.nan_to_num(X_std)
    return X_std

def perform_factor_analysis(data, dimensions, n_factors=1):
    data_clean = data[dimensions].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data_clean) == 0:
        return None
    X_std = safe_standardize(data_clean)
    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    try:
        return fa.fit_transform(X_std)
    except Exception as e:
        logging.error(f"Error in factor analysis: {e}")
        return None

def calculate_latent_scores(df, dimensions, group_cols=['unique_id', 'model_name', 'initialization']):
    latent_scores = df.groupby(group_cols).apply(
        lambda x: perform_factor_analysis(x, dimensions)
    )
    return latent_scores

def compare_mean_latent_scores(df, dimensions):
    true_latent = calculate_latent_scores(
        df[df['model_name'] == 'true'],
        dimensions,
        group_cols=['unique_id']
    )

    model_latent_scores = {}
    correlations = {}

    for model in df['model_name'].unique():
        if model == 'true':
            continue

        model_latent = calculate_latent_scores(
            df[df['model_name'] == model],
            dimensions
        )

        # Calculate mean latent scores across initializations
        mean_latent = model_latent.groupby('unique_id').mean()

        model_latent_scores[model] = mean_latent

        # Align true and model latent scores
        aligned_true = true_latent.loc[mean_latent.index]

        # Calculate correlation
        corr, _ = pearsonr(aligned_true.flatten(), mean_latent.values.flatten())
        correlations[model] = corr

    return correlations, model_latent_scores, true_latent

def generate_dimension_summary(per_model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Generate a summary of performance across dimensions."""
    dimension_summary = {}
    try:
        dimensions = set()
        for model_results in per_model_results.values():
            dimensions.update(model_results['aggregated']['dimension_metrics'].keys())

        for dimension in dimensions:
            dimension_summary[dimension] = {
                "best_model": None,
                "worst_model": None,
                "average_performance": {metric: [] for metric in next(iter(per_model_results.values()))['aggregated']['dimension_metrics'][dimension]}
            }

            for model, results in per_model_results.items():
                metrics = results['aggregated']['dimension_metrics'][dimension]

                if dimension_summary[dimension]["best_model"] is None or \
                   (metrics['accuracy'] > per_model_results[dimension_summary[dimension]["best_model"]]['aggregated']['dimension_metrics'][dimension]['accuracy']):
                    dimension_summary[dimension]["best_model"] = model

                if dimension_summary[dimension]["worst_model"] is None or \
                   (metrics['accuracy'] < per_model_results[dimension_summary[dimension]["worst_model"]]['aggregated']['dimension_metrics'][dimension]['accuracy']):
                    dimension_summary[dimension]["worst_model"] = model

                for metric, value in metrics.items():
                    dimension_summary[dimension]["average_performance"][metric].append(value)

        # Calculate averages
        for dimension in dimension_summary:
            for metric in dimension_summary[dimension]["average_performance"]:
                dimension_summary[dimension]["average_performance"][metric] = np.mean(dimension_summary[dimension]["average_performance"][metric])

    except Exception as e:
        logging.error(f"Error generating dimension summary: {e}")
        raise
    return dimension_summary

def aggregate_initialization_results(init_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results across initializations for a single model."""
    aggregated = {
        'dimension_metrics': {},
        'aggregate_metrics': {},
        'cost_metrics': {}
    }

    # Aggregate dimension metrics
    for dimension in init_results[0]['dimension_metrics']:
        aggregated['dimension_metrics'][dimension] = {}
        for metric in init_results[0]['dimension_metrics'][dimension]:
            values = [r['dimension_metrics'][dimension][metric] for r in init_results]
            aggregated['dimension_metrics'][dimension][metric] = np.mean(values)

    # Aggregate aggregate metrics
    for metric in init_results[0]['aggregate_metrics']:
        values = [r['aggregate_metrics'][metric] for r in init_results]
        aggregated['aggregate_metrics'][metric] = np.mean(values)

    # Aggregate cost metrics
    for metric in init_results[0]['cost_metrics']:
        values = [r['cost_metrics'][metric] for r in init_results]
        aggregated['cost_metrics'][metric] = np.mean(values)

    return aggregated

def main(config_path: str):
    try:
        config = load_config(config_path)
        setup_logging(config, "evaluate_models")

        logging.info("Starting evaluation")

        df = load_data(config)
        logging.info(f"Loaded data shape: {df.shape}")
        logging.info(f"Columns in loaded data: {df.columns.tolist()}")

        models = get_model_names(config)
        logging.info(f"Evaluating models: {', '.join(models)}")

        per_model_results = {}
        confusion_matrices = {}

        for model in tqdm(models, desc="Evaluating models"):
            model_config = next((m for m in config['models'] if m['name'] == model), None)
            if not model_config:
                logging.warning(f"Model {model} not found in config")
                continue

            n_random_init = model_config['prompt'].get('n_random_init', 1)
            model_init_results = []
            model_init_cms = []

            for init in range(n_random_init):
                try:
                    metrics, cm = calculate_metrics(df, model, init, config)

                    aggregate_metrics = calculate_aggregate_metrics(metrics, config)

                    cost_metrics = calculate_cost_metrics(df, model, init)

                    init_results = {
                        'dimension_metrics': metrics,
                        'aggregate_metrics': aggregate_metrics,
                        'cost_metrics': cost_metrics
                    }
                    model_init_results.append(init_results)
                    model_init_cms.append(cm)
                except Exception as e:
                    logging.error(f"Error processing {model} (init {init}): {e}")

            if model_init_results:
                # Aggregate results across initializations
                aggregated_results = aggregate_initialization_results(model_init_results)

                per_model_results[model] = {
                    'aggregated': aggregated_results,
                    'initializations': {init: results for init, results in enumerate(model_init_results)}
                }

                confusion_matrices[model] = {
                    'initializations': {init: cms for init, cms in enumerate(model_init_cms)}
                }
            else:
                logging.warning(f"No valid results for model {model}")

        try:
            significance_tests = perform_significance_tests(df, models, config)
        except Exception as e:
            logging.error(f"Error performing significance tests: {e}")
            significance_tests = {}

        try:
            dimension_summary = generate_dimension_summary(per_model_results)
        except Exception as e:
            logging.error(f"Error generating dimension summary: {e}")
            dimension_summary = {}

        # Perform factor analysis
        dimensions = [
            'Participation',
            'Justification-level',
            'Justification-content',
            'Respect-group',
            'Respect-demand',
            'Respect-counterarg',
            'Constructive Politics'
        ]

        try:
            latent_correlations, model_latent_scores, true_latent = compare_mean_latent_scores(df, dimensions)

            # Add individual dimension correlations
            for dim in dimensions:
                for model in latent_correlations.keys():
                    model_pred = df[df['model_name'] == model][f"{dim}_pred"]
                    true_val = df[df['model_name'] == 'true'][f"{dim}_true"]
                    corr, _ = pearsonr(model_pred, true_val)
                    latent_correlations[model + '_' + dim] = corr

            latent_analysis = {
                "correlations": latent_correlations,
                "model_latent_scores": {model: scores.to_dict() for model, scores in model_latent_scores.items()},
                "true_latent_scores": true_latent.to_dict()
            }
        except Exception as e:
            logging.error(f"Error performing latent analysis: {e}")
            latent_analysis = {}

        results = {
            "metadata": {
                "date_generated": datetime.now().isoformat(),
                "config_used": config,
                "models_evaluated": models
            },
            "per_model_results": per_model_results,
            "significance_tests": significance_tests,
            "dimension_summary": dimension_summary,
            "latent_analysis": latent_analysis
        }

        if config['analysis'].get('export_confusion_matrices', True):
            results["confusion_matrices"] = confusion_matrices

        export_results(results, config)

        logging.info("Evaluation complete.")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except KeyError as e:
        logging.error(f"Missing configuration key: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during evaluation: {str(e)}", exc_info=True)
    finally:
        logging.info("Evaluation script finished execution.")

def export_results(results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Export analysis results to JSON format."""
    try:
        output_dir = config['data']['results_directory']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        json_output = os.path.join(output_dir, 'evaluation_results.json')
        with open(json_output, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        logging.info(f"Results exported to JSON: {json_output}")

    except Exception as e:
        logging.error(f"Error exporting results: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument("--config", help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)
