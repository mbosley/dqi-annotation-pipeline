#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import argparse
import logging
from typing import Dict, Any, List, Tuple

def load_config(config_file: str = 'config.yaml') -> Dict[str, Any]:
    import yaml
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def setup_logging(config: Dict[str, Any]) -> None:
    log_config = config['logging']
    logging.basicConfig(
        level=logging.getLevelName(config['global']['log_level']),
        format=log_config['format'],
        datefmt=log_config['date_format'],
        filename=os.path.join(log_config['directory'], 'visualize_results.log'),
        filemode='w'
    )

def load_data(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        return json.load(f)

def ensure_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_model_performance(data: Dict[str, Any], metric: str, output_dir: str) -> None:
    output_file = os.path.join(output_dir, f'model_performance_{metric}.png')

    try:
        models = list(data['per_model_results'].keys())
        if not models:
            logging.warning(f"No models found in the data for metric: {metric}")
            return

        dimensions = list(data['per_model_results'][models[0]]['aggregated']['dimension_metrics'].keys())
        if not dimensions:
            logging.warning(f"No dimensions found in the data for metric: {metric}")
            return

        performance = {}
        for model in models:
            model_data = data['per_model_results'][model]['aggregated']['dimension_metrics']
            performance[model] = [model_data[dim].get(metric, np.nan) for dim in dimensions]

        df = pd.DataFrame(performance, index=dimensions)

        if df.empty:
            logging.warning(f"DataFrame is empty for metric: {metric}")
            return

        plt.figure(figsize=(12, 8))
        ax = df.plot(kind='bar', width=0.8)
        plt.title(f'Model Performance Comparison ({metric.capitalize()})')
        plt.xlabel('Dimensions')
        plt.ylabel(metric.capitalize())
        plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

        logging.info(f"Saved model performance plot for {metric} to {output_file}")

    except KeyError as e:
        logging.error(f"KeyError in plot_model_performance for metric {metric}: {str(e)}")
    except Exception as e:
        logging.error(f"Error in plot_model_performance for metric {metric}: {str(e)}")

def plot_confusion_matrices(data: Dict[str, Any], output_dir: str) -> None:
    confusion_matrices_dir = os.path.join(output_dir, 'confusion_matrices')
    ensure_directory(confusion_matrices_dir)

    for model, matrices in data['confusion_matrices'].items():
        for init, init_matrices in matrices['initializations'].items():
            for dimension, matrix_data in init_matrices.items():
                output_file = os.path.join(confusion_matrices_dir, f'confusion_matrix_{model}_init{init}_{dimension}.png')
                matrix = np.array(matrix_data['matrix'])
                labels = matrix_data['labels']

                plt.figure(figsize=(8, 6))
                sns.heatmap(matrix, annot=True, fmt='d', cmap='YlOrRd',
                            xticklabels=labels, yticklabels=labels)
                plt.title(f'Confusion Matrix: {model} - Init {init} - {dimension}')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.tight_layout()
                plt.savefig(output_file)
                plt.close()

    logging.info(f"Saved confusion matrices to {confusion_matrices_dir}")

def plot_radar_chart(data: Dict[str, Any], output_dir: str) -> None:
    output_file = os.path.join(output_dir, 'radar_chart.png')

    models = list(data['per_model_results'].keys())
    metrics = ['accuracy', 'f1', 'oci']

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    for model in models:
        values = []
        for metric in metrics:
            values.append(np.mean([data['per_model_results'][model]['aggregated']['dimension_metrics'][dim][metric]
                                   for dim in data['per_model_results'][model]['aggregated']['dimension_metrics']]))
        values = np.concatenate((values, [values[0]]))

        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.25)

    ax.set_thetagrids(angles[:-1] * 180/np.pi, metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Across Metrics")
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    logging.info(f"Saved radar chart to {output_file}")

def plot_cost_comparison(data: Dict[str, Any], output_dir: str) -> None:
    output_file = os.path.join(output_dir, 'cost_comparison.png')

    models = list(data['per_model_results'].keys())
    total_costs = [data['per_model_results'][model]['aggregated']['cost_metrics']['total_cost'] for model in models]
    avg_costs = [data['per_model_results'][model]['aggregated']['cost_metrics']['average_cost'] for model in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, total_costs, width, label='Total Cost')
    ax.bar(x + width/2, avg_costs, width, label='Average Cost')

    ax.set_ylabel('Cost')
    ax.set_title('Cost Comparison Across Models')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    logging.info(f"Saved cost comparison plot to {output_file}")

def plot_performance_vs_cost(data: Dict[str, Any], output_dir: str) -> None:
    output_file = os.path.join(output_dir, 'performance_vs_cost.png')

    models = list(data['per_model_results'].keys())
    avg_costs = [data['per_model_results'][model]['aggregated']['cost_metrics']['average_cost'] for model in models]
    accuracies = [np.mean([data['per_model_results'][model]['aggregated']['dimension_metrics'][dim]['accuracy']
                           for dim in data['per_model_results'][model]['aggregated']['dimension_metrics']])
                  for model in models]

    # Filter out models with zero cost
    filtered_models = [model for model, cost in zip(models, avg_costs) if cost < 0.3]
    filtered_costs = [cost for cost in avg_costs if cost < 0.3]
    filtered_accuracies = [acc for acc, cost in zip(accuracies, avg_costs) if cost < 0.3]

    # Create model families
    model_families = {}
    for model in filtered_models:
        family = '-'.join(model.split('-')[:-1])  # Assuming the last part is ICL count
        if family not in model_families:
            model_families[family] = []
        model_families[family].append(model)

    # Set up plot
    plt.figure(figsize=(5, 4))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(model_families)))

    for (family, models), color in zip(model_families.items(), colors):
        family_costs = [filtered_costs[filtered_models.index(model)] for model in models]
        family_accuracies = [filtered_accuracies[filtered_models.index(model)] for model in models]

        plt.scatter(family_costs, family_accuracies, c=[color], label=family)

        # Add lines connecting the points
        plt.plot(family_costs, family_accuracies, c=color, alpha=0.5)

        # Add labels for each point
        for model, cost, acc in zip(models, family_costs, family_accuracies):
            plt.annotate(model.split('-')[-1], (cost, acc), xytext=(5, 5),
                         textcoords='offset points', alpha=0.8)

    plt.xscale('log')  # Use log scale for x-axis
    plt.xlabel('Average Cost per Query (Log Scale)')
    plt.ylabel('Average Accuracy')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

    logging.info(f"Saved performance vs cost plot to {output_file}")

def plot_dimension_heatmap(data: Dict[str, Any], output_dir: str) -> None:
    output_file = os.path.join(output_dir, 'dimension_heatmap.png')

    models = list(data['per_model_results'].keys())
    dimensions = list(data['per_model_results'][models[0]]['aggregated']['dimension_metrics'].keys())

    performance = {model: [data['per_model_results'][model]['aggregated']['dimension_metrics'][dim]['accuracy'] for dim in dimensions]
                   for model in models}

    df = pd.DataFrame(performance, index=dimensions)

    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.3f')
    plt.title('Dimension-wise Performance Heatmap (Accuracy)')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    logging.info(f"Saved dimension heatmap to {output_file}")

def plot_significance_heatmap(data: Dict[str, Any], output_dir: str) -> None:
    output_file = os.path.join(output_dir, 'significance_heatmap.png')

    t_tests = data['significance_tests']['t_tests']
    dimensions = list(t_tests.keys())
    models = list(t_tests[dimensions[0]].keys())

    df = pd.DataFrame({dim: [t_tests[dim][model] for model in models] for dim in dimensions}, index=models)

    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, cmap='YlOrRd', fmt='.3e')
    plt.title('Significance Test Heatmap (p-values)')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    logging.info(f"Saved significance heatmap to {output_file}")

def plot_initialization_comparison(data: Dict[str, Any], output_dir: str) -> None:
    output_file = os.path.join(output_dir, 'initialization_comparison.png')

    models = list(data['per_model_results'].keys())
    metrics = ['accuracy', 'f1', 'oci']

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 6*len(metrics)), sharex=True)
    fig.suptitle('Model Performance Across Initializations')

    for i, metric in enumerate(metrics):
        for model in models:
            init_results = data['per_model_results'][model]['initializations']
            x = list(range(len(init_results)))
            y = [init_results[str(init)]['aggregate_metrics'][f'avg_{metric}'] for init in x]
            axes[i].plot(x, y, 'o-', label=model)

        axes[i].set_ylabel(metric.upper())
        axes[i].set_xlabel('Initialization')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    logging.info(f"Saved initialization comparison plot to {output_file}")

def plot_initialization_boxplots(data: Dict[str, Any], output_dir: str) -> None:
    output_file = os.path.join(output_dir, 'initialization_boxplots.png')

    models = list(data['per_model_results'].keys())
    metrics = ['accuracy', 'f1', 'oci']

    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6))
    fig.suptitle('Distribution of Metrics Across Initializations')

    for i, metric in enumerate(metrics):
        metric_data = []
        for model in models:
            init_results = data['per_model_results'][model]['initializations']
            metric_values = [init_results[str(init)]['aggregate_metrics'][f'avg_{metric}'] for init in range(len(init_results))]
            metric_data.append(metric_values)

        axes[i].boxplot(metric_data, labels=models)
        axes[i].set_title(metric.upper())
        axes[i].set_ylabel('Value')
        axes[i].set_xlabel('Model')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    logging.info(f"Saved initialization boxplots to {output_file}")

def plot_performance_vs_icl(data: Dict[str, Any], output_dir: str, metric: str = 'accuracy',
                            fig_size: Tuple[int, int] = (9, 6), title: str = None) -> None:
    output_file = os.path.join(output_dir, f'performance_vs_icl_{metric}.png')

    try:
        # Group models by base name (without ICL count)
        model_groups = {}
        for model in data['per_model_results'].keys():
            base_name = '-'.join(model.split('-')[:-1])
            icl_count = int(model.split('-')[-1])
            if base_name not in model_groups:
                model_groups[base_name] = []
            model_groups[base_name].append((icl_count, model))

        plt.figure(figsize=fig_size)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(model_groups)))

        for (base_name, models), color in zip(model_groups.items(), colors):
            icl_examples = []
            performances = []

            # Sort models by ICL count
            for icl_count, model in sorted(models):
                icl_examples.append(icl_count)
                avg_performance = np.mean([
                    data['per_model_results'][model]['aggregated']['dimension_metrics'][dim][metric]
                    for dim in data['per_model_results'][model]['aggregated']['dimension_metrics']
                ])
                performances.append(avg_performance)

            # Plot line for this model group
            plt.plot(icl_examples, performances, '-o', color=color, label=base_name.replace('-icl', ''))

        plt.xlabel('Number of ICL Examples')
        if metric.lower() in ['mse', 'mae']:
            plt.ylabel(f'Average {metric.upper()}')
        else:
            plt.ylabel(f'Average {metric.capitalize()}')
        # plt.title(title or f'Model Performance ({metric.upper() if metric.lower() in ["mse", "mae"] else metric.capitalize()}) vs Number of ICL Examples')

        # Customize legend
        legend = plt.legend(title="Models", fontsize='large', frameon=True, framealpha=0.8)
        legend.get_title().set_fontsize('large')  # Set title font size

        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

        logging.info(f"Saved performance vs ICL examples plot for {metric} to {output_file}")

    except KeyError as e:
        logging.error(f"KeyError in plot_performance_vs_icl: {str(e)}")
    except ValueError as e:
        logging.error(f"ValueError in plot_performance_vs_icl: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error in plot_performance_vs_icl: {str(e)}")

def plot_performance_vs_icl_by_dimension(data: Dict[str, Any], output_dir: str, metric: str = 'accuracy',
                                         fig_size: Tuple[int, int] = (12, 8), title: str = None, legend: bool = False,
                                         title_on: bool = False) -> None:
    if metric.lower() == 'mse':
        metric_title = 'MSE'
    elif metric.lower() == 'mae':
        metric_title = 'MAE'
    else:
        metric_title = metric.capitalize()

    output_file = os.path.join(output_dir, f'performance_vs_icl_{metric}_by_dimension.png')

    try:
        # Group models by base name (without ICL count)
        model_groups = {}
        for model in data['per_model_results'].keys():
            base_name = '-'.join(model.split('-')[:-1])
            icl_count = int(model.split('-')[-1])
            if base_name not in model_groups:
                model_groups[base_name] = []
            model_groups[base_name].append((icl_count, model))

        dimensions = list(data['per_model_results'][list(data['per_model_results'].keys())[0]]['aggregated']['dimension_metrics'].keys())

        fig, axes = plt.subplots(2, 4, figsize=fig_size)

        if title_on:
            fig.suptitle(title or f'Model Performance ({metric_title}) vs Number of ICL Examples by Dimension')

        axes = axes.flatten()

        colors = plt.cm.rainbow(np.linspace(0, 1, len(model_groups)))

        for idx, dimension in enumerate(dimensions):
            ax = axes[idx]

            for (base_name, models), color in zip(model_groups.items(), colors):
                icl_examples = []
                performances = []

                # Sort models by ICL count
                for icl_count, model in sorted(models):
                    icl_examples.append(icl_count)
                    performance = data['per_model_results'][model]['aggregated']['dimension_metrics'][dimension][metric]
                    performances.append(performance)

                # Plot line for this model group
                ax.plot(icl_examples, performances, '-o', color=color, label=base_name.replace('-icl', ''))

            ax.set_xlabel('Number of ICL Examples')
            ax.set_ylabel(metric_title)
            ax.set_title(dimension)
            ax.grid(True, linestyle='--', alpha=0.7)

        for idx in range(len(dimensions), len(axes)):
            ax = axes[idx]
            ax.axis('off')  # Turn off the axis for empty subplots

        if legend:
            # Use the last subplot for the legend
            legend_ax = axes[-1]
            legend_ax.axis('off')
            handles, labels = axes[0].get_legend_handles_labels()
            legend_ax.legend(handles, labels, loc='center', title='Models')

        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

        logging.info(f"Saved performance vs ICL examples plot by dimension for {metric} to {output_file}")

    except Exception as e:
        logging.error(f"Error in plot_performance_vs_icl_by_dimension: {str(e)}")

def plot_radar_charts_by_model_group(data: Dict[str, Any], output_dir: str, metric: str = 'accuracy',
                                     fig_size: Tuple[int, int] = (20, 15), title: str = None) -> None:
    output_file = os.path.join(output_dir, f'radar_charts_{metric}_by_model_group.png')

    try:
        # Group models by base name (without ICL count)
        model_groups = {}
        for model in data['per_model_results'].keys():
            base_name = '-'.join(model.split('-')[:-1])
            icl_count = int(model.split('-')[-1])
            if base_name not in model_groups:
                model_groups[base_name] = []
            model_groups[base_name].append((icl_count, model))

        dimensions = list(data['per_model_results'][list(data['per_model_results'].keys())[0]]['aggregated']['dimension_metrics'].keys())

        num_groups = len(model_groups)
        rows = (num_groups + 1) // 2  # Calculate number of rows needed
        fig, axes = plt.subplots(rows, 2, figsize=fig_size, subplot_kw=dict(projection='polar'))
        fig.suptitle(title or f'Model Performance ({metric.capitalize()}) by Dimension for Each Model Group')

        axes = axes.flatten()

        for idx, (base_name, models) in enumerate(model_groups.items()):
            ax = axes[idx]

            # Prepare data for radar chart
            angles = np.linspace(0, 2*np.pi, len(dimensions), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))  # Complete the polygon

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(dimensions)
            ax.set_title(base_name.replace('-icl', ''))

            for icl_count, model in models:
                values = [data['per_model_results'][model]['aggregated']['dimension_metrics'][dim][metric] for dim in dimensions]
                values = np.concatenate((values, [values[0]]))  # Complete the polygon

                ax.plot(angles, values, 'o-', linewidth=2, label=f'ICL-{icl_count}')
                ax.fill(angles, values, alpha=0.25)

            ax.set_ylim(0, 1)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        # Remove any unused subplots
        for idx in range(len(model_groups), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()

        logging.info(f"Saved radar charts by model group for {metric} to {output_file}")

    except Exception as e:
        logging.error(f"Error in plot_radar_charts_by_model_group: {str(e)}")

def main(config: Dict[str, Any]) -> None:
    setup_logging(config)
    logging.info("Starting visualization process")

    results_file = os.path.join(config['data']['results_directory'], 'evaluation_results.json')
    try:
        data = load_data(results_file)
    except FileNotFoundError:
        logging.error(f"Results file not found: {results_file}")
        return
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in results file: {results_file}")
        return

    output_dir = os.path.join(config['data']['results_directory'], 'figures')
    ensure_directory(output_dir)

    vis_config = config['analysis'].get('visualizations', {})

    # Generate visualizations based on config settings
    if vis_config['generate_bar_plots']:
        logging.info("Generating bar plots for model performance")
        for metric in config['analysis']['metrics_to_analyze']:
            plot_model_performance(data, metric, output_dir)

    if config['analysis']['export_confusion_matrices']:
        logging.info("Generating confusion matrices")
        plot_confusion_matrices(data, output_dir)

    if vis_config['generate_radar_charts']:
        logging.info("Generating radar charts")
        plot_radar_chart(data, output_dir)

    if vis_config['generate_cost_comparison']:
        logging.info("Generating cost comparison plot")
        plot_cost_comparison(data, output_dir)

    if vis_config['generate_performance_vs_cost']:
        logging.info("Generating performance vs cost plot")
        plot_performance_vs_cost(data, output_dir)

    if vis_config['generate_dimension_heatmap']:
        logging.info("Generating dimension heatmap")
        plot_dimension_heatmap(data, output_dir)

    if vis_config['generate_significance_heatmap']:
        logging.info("Generating significance heatmap")
        plot_significance_heatmap(data, output_dir)

    if vis_config.get('generate_initialization_comparison', False):
        logging.info("Generating initialization comparison plot")
        plot_initialization_comparison(data, output_dir)

    if config['analysis']['initialization_analysis'].get('generate_boxplots', False):
        logging.info("Generating initialization boxplots")
        plot_initialization_boxplots(data, output_dir)

    if vis_config.get('generate_performance_vs_icl', True):
        logging.info("Generating performance vs ICL examples plots")
        plot_performance_vs_icl(data, output_dir, metric="accuracy")
        plot_performance_vs_icl(data, output_dir, metric="f1")
        plot_performance_vs_icl(data, output_dir, metric="oci")
        plot_performance_vs_icl(data, output_dir, metric="mse")
        plot_performance_vs_icl(data, output_dir, metric="mae")

        logging.info("Generating performance vs ICL examples plots by dimension")
        plot_performance_vs_icl_by_dimension(data, output_dir, metric="accuracy")
        plot_performance_vs_icl_by_dimension(data, output_dir, metric="f1")
        plot_performance_vs_icl_by_dimension(data, output_dir, metric="oci")
        plot_performance_vs_icl_by_dimension(data, output_dir, metric="mse")
        plot_performance_vs_icl_by_dimension(data, output_dir, metric="mae")

        logging.info("Generating radar charts by model group")
        plot_radar_charts_by_model_group(data, output_dir, metric="accuracy")
        plot_radar_charts_by_model_group(data, output_dir, metric="f1")
        plot_radar_charts_by_model_group(data, output_dir, metric="oci")
        plot_radar_charts_by_model_group(data, output_dir, metric="mse")

    logging.info("Visualization process completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
