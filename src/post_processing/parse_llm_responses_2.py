#!/usr/bin/env python3
import json
import csv
import argparse
import logging
import yaml
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add a file handler for detailed logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
file_handler = logging.FileHandler(log_dir / 'llm_parsing.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

def load_config(config_file: str = 'config.yaml') -> Dict[str, Any]:
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def extract_score_data(content: str, top_logprobs: int) -> List[Dict[str, Any]]:
    score_patterns = re.finditer(r'"score":\s*(\[?\d+\]?)', content)
    results = []
    for match in score_patterns:
        score_str = match.group(1)
        # Handle both [1] and 1 cases
        score = int(score_str.strip('[]'))
        token_start = content.rfind('"token":', 0, match.start())
        if token_start == -1:
            continue
        section = content[token_start:match.start()]
        try:
            token_data = json.loads('{' + section.strip().rstrip(',') + '}')
        except json.JSONDecodeError:
            continue
        logprobs = token_data.get('top_logprobs', [])
        results.append({
            'score': score,
            'top_logprobs': [{
                'token': item['token'],
                'logprob': item['logprob']
            } for item in logprobs if item['token'].strip().isdigit()][:top_logprobs]
        })
    return results

def parse_jsonl(input_file: str, output_file: str, model_name: str, init: int, config: Dict[str, Any], include_text: bool = False, include_metadata: bool = False, include_logprobs: bool = False):
    model_config = next((model for model in config['models'] if model['name'] == model_name), None)
    top_logprobs = model_config['model_params'].get('top_logprobs', 0)
    logprobs_enabled = model_config['model_params'].get('logprobs', False)

    successful_entries = 0
    failed_entries = 0

    def clean_score(score):
        if isinstance(score, list):
            logger.warning(f"Found score as list: {score}. Using first element.")
            return score[0] if score else 0
        return score

    with open(input_file, 'r') as jsonl_file, open(output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Define headers
        base_header = ['unique_id', 'model_name', 'initialization']
        score_header = [
            'participation_score',
            'justification_level_score',
            'justification_content_score',
            'respect_groups_score',
            'respect_demand_score',
            'respect_counterargument_score',
            'constructive_politics_score'
        ]

        # Add headers for logprobs if enabled and requested
        logprob_header = []
        if include_logprobs and logprobs_enabled and top_logprobs > 0:
            for score_name in score_header:
                logprob_header.extend([f'{score_name}_logprob{i+1}' for i in range(top_logprobs)])

        text_header = [
            'participation_reasoning',
            'justification_level_reasoning',
            'justification_content_reasoning',
            'respect_groups_reasoning',
            'respect_demand_reasoning',
            'respect_counterargument_reasoning',
            'constructive_politics_reasoning'
        ] if include_text else []

        metadata_header = [
            'prompt_tokens', 'completion_tokens', 'total_tokens',
            'latency', 'estimated_cost', 'model_version', 'healing_attempts',
            'healing_success', 'healing_method'
        ] if include_metadata else []

        # Combine all headers in the desired order
        header = base_header + score_header + logprob_header + text_header + metadata_header

        if include_text:
            header.insert(3, 'prompt')  # Add 'prompt' right after 'initialization' if text is included

        csv_writer.writerow(header)

        for line_number, line in enumerate(jsonl_file, 1):
            try:
                data = json.loads(line)
                logger.debug(f"Processing entry {line_number} with unique_id: {data['original_item'].get('unique_id', 'Unknown')}")

                if 'original_item' not in data or 'response' not in data['original_item']:
                    raise KeyError("Missing 'original_item' or 'response' key")

                # Parse the response, handling the case where it might already be a dictionary
                if isinstance(data['original_item']['response'], str):
                    response = json.loads(data['original_item']['response'])
                else:
                    response = data['original_item']['response']

                response = data['healed_json'] if data['healing_success'] else response

                # Check if 'annotation' is at the top level, if not, assume the entire response is the annotation
                annotation = response.get('annotation', response)

                row = [data['original_item']['unique_id'], model_name, init]

                if include_text:
                    row.append(data['original_item']['prompt'])

                scores = [
                    clean_score(annotation['participation']['score']),
                    clean_score(annotation['justification']['level']['score']),
                    clean_score(annotation['justification']['content']['score']),
                    clean_score(annotation['respect']['groups']['score']),
                    clean_score(annotation['respect']['demand']['score']),
                    clean_score(annotation['respect']['counterargument']['score']),
                    clean_score(annotation['constructive_politics']['score'])
                ]
                row.extend(scores)

                # Extract and add logprobs if enabled and requested
                if include_logprobs and logprobs_enabled and top_logprobs > 0:
                    score_data = extract_score_data(json.dumps(data['original_item']['response']), top_logprobs)
                    for score_info in score_data:
                        logprobs = score_info['top_logprobs']
                        row.extend([lp['logprob'] for lp in logprobs] + [''] * (top_logprobs - len(logprobs)))  # Pad with empty strings if less than top_logprobs

                if include_text:
                    row.extend([
                        annotation['participation']['reasoning'],
                        annotation['justification']['level']['reasoning'],
                        annotation['justification']['content']['reasoning'],
                        annotation['respect']['groups']['reasoning'],
                        annotation['respect']['demand']['reasoning'],
                        annotation['respect']['counterargument']['reasoning'],
                        annotation['constructive_politics']['reasoning']
                    ])

                if include_metadata:
                    row.extend([
                        data['original_item'].get('prompt_tokens', ''),
                        data['original_item'].get('completion_tokens', ''),
                        data['original_item'].get('total_tokens', ''),
                        data['original_item'].get('latency', ''),
                        data['original_item'].get('estimated_cost', ''),
                        data['original_item'].get('model_version', ''),
                        data.get('healing_attempts', ''),
                        data['healing_success'],
                        data.get('healing_method', '')
                    ])

                csv_writer.writerow(row)
                successful_entries += 1

            except json.JSONDecodeError as e:
                failed_entries += 1
                logger.error(f"JSON decoding error in line {line_number}: {e}")
                logger.debug(f"Problematic line content: {line[:200]}...")  # Log first 200 characters
            except KeyError as e:
                failed_entries += 1
                logger.error(f"Key error in line {line_number}: {e}")
                logger.debug(f"Data structure: {json.dumps(data, indent=2)}")
            except Exception as e:
                failed_entries += 1
                logger.error(f"Unexpected error in line {line_number}: {e}")
                logger.debug(f"Data structure: {json.dumps(data, indent=2)}")

    logger.info(f"Processing complete. Successful entries: {successful_entries}, Failed entries: {failed_entries}")
    if failed_entries > 0:
        logger.warning(f"Failed to process {failed_entries} entries. Check the log for details.")

def main():
    parser = argparse.ArgumentParser(description="Parse JSONL responses and generate CSV")
    parser.add_argument("--model", required=True, help="Model name to process responses for")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--include-text", action="store_true", help="Include text-heavy fields in the output")
    parser.add_argument("--include-metadata", action="store_true", help="Include metadata fields in the output")
    parser.add_argument("--include-logprobs", action="store_true", help="Include logprobs in the output if available")
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = next((model for model in config['models'] if model['name'] == args.model), None)
    if not model_config:
        raise ValueError(f"Model {args.model} not found in config")

    n_random_init = model_config['prompt'].get('n_random_init', 1)

    for init in range(n_random_init):
        input_file = Path(config['data']['api_response_directory']) / f"api_responses_{args.model}_init-{init}_healed.jsonl"
        output_file = Path(config['data']['results_directory']) / f"processed_results_{args.model}_init-{init}.csv"

        logger.info(f"Processing responses for model: {args.model}, initialization: {init}")
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Including text-heavy fields: {args.include_text}")
        logger.info(f"Including metadata fields: {args.include_metadata}")
        logger.info(f"Including logprobs: {args.include_logprobs}")

        parse_jsonl(str(input_file), str(output_file), args.model, init, config, args.include_text, args.include_metadata, args.include_logprobs)
        logger.info(f"Parsing complete for initialization {init}. Results saved to {output_file}")

if __name__ == "__main__":
    main()
