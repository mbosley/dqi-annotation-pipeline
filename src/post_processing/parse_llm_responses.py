#!/usr/bin/env python3
import json
import csv
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List

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

def parse_jsonl(input_file: str, output_file: str, model_name: str, include_text: bool = False, include_metadata: bool = False):
    with open(input_file, 'r') as jsonl_file, open(output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Define headers
        base_header = ['unique_id', 'model_name']
        score_header = [
            'participation_score',
            'justification_level_score',
            'justification_content_score',
            'respect_groups_score',
            'constructive_politics_score'
        ]

        text_header = [
            'participation_reasoning',
            'justification_level_reasoning',
            'justification_content_reasoning',
            'respect_groups_reasoning',
            'constructive_politics_reasoning',
            'notes', 'summary'
        ] if include_text else []

        metadata_header = [
            'prompt_tokens', 'completion_tokens', 'total_tokens',
            'latency', 'estimated_cost', 'model_version', 'healing_attempts',
            'healing_success', 'healing_method'
        ] if include_metadata else []

        # Dynamic headers for demands and counterarguments
        max_demands = 0
        max_counterarguments = 0

        # First pass to determine the maximum number of demands and counterarguments
        for line in jsonl_file:
            try:
                data = json.loads(line)
                response = data['healed_json'] if data['healing_success'] else json.loads(data['original_item']['response']['content'][0]['text'])
                max_demands = max(max_demands, len(response['respect']['demands']))
                max_counterarguments = max(max_counterarguments, len(response['respect']['counterarguments']))
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON in first pass: {line[:50]}...")
            except KeyError:
                logger.warning(f"Skipping entry with unexpected structure in first pass: {line[:50]}...")
            except Exception as e:
                logger.warning(f"Unexpected error in first pass: {str(e)}. Skipping entry: {line[:50]}...")

        # Create dynamic headers
        dynamic_header = []
        for i in range(max_demands):
            dynamic_header.append(f'respect_demand_{i+1}_score')
            if include_text:
                dynamic_header.append(f'respect_demand_{i+1}_reasoning')
        for i in range(max_counterarguments):
            dynamic_header.append(f'respect_counterargument_{i+1}_score')
            if include_text:
                dynamic_header.append(f'respect_counterargument_{i+1}_reasoning')

        # Combine all headers in the desired order
        header = base_header + score_header + dynamic_header + text_header + metadata_header

        if include_text:
            header.insert(2, 'prompt')  # Add 'prompt' right after 'model_name' if text is included

        csv_writer.writerow(header)

        # Reset file pointer for second pass
        jsonl_file.seek(0)

        for line in jsonl_file:
            try:
                data = json.loads(line)
                response = data['healed_json'] if data['healing_success'] else json.loads(data['original_item']['response']['content'][0]['text'])

                row = [data['original_item']['unique_id'], model_name]

                if include_text:
                    row.append(data['original_item']['prompt'])

                row.extend([
                    response['participation']['score'],
                    response['justification']['level']['score'],
                    response['justification']['content']['score'],
                    response['respect']['groups']['score'],
                    response['constructive_politics']['score']
                ])

                # Add demands data
                for i in range(max_demands):
                    if i < len(response['respect']['demands']):
                        row.append(response['respect']['demands'][i]['score'])
                        if include_text:
                            row.append(response['respect']['demands'][i]['reasoning'])
                    else:
                        row.append('')
                        if include_text:
                            row.append('')

                # Add counterarguments data
                for i in range(max_counterarguments):
                    if i < len(response['respect']['counterarguments']):
                        row.append(response['respect']['counterarguments'][i]['score'])
                        if include_text:
                            row.append(response['respect']['counterarguments'][i]['reasoning'])
                    else:
                        row.append('')
                        if include_text:
                            row.append('')

                if include_text:
                    row.extend([
                        response['participation']['reasoning'],
                        response['justification']['level']['reasoning'],
                        response['justification']['content']['reasoning'],
                        response['respect']['groups']['reasoning'],
                        response['constructive_politics']['reasoning'],
                        response.get('notes', ''),
                        response.get('summary', '')
                    ])

                if include_metadata:
                    row.extend([
                        data['original_item'].get('prompt_tokens', ''),
                        data['original_item'].get('completion_tokens', ''),
                        data['original_item'].get('total_tokens', ''),
                        data['original_item'].get('latency', ''),
                        data['original_item'].get('estimated_cost', ''),
                        data['original_item'].get('model_version', ''),
                        data['healing_attempts'],
                        data['healing_success'],
                        data['healing_method']
                    ])

                csv_writer.writerow(row)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding error for unique_id {data['original_item'].get('unique_id', 'Unknown')}: {e}. Skipping entry.")
            except KeyError as e:
                logger.error(f"Key error when processing response for unique_id {data['original_item'].get('unique_id', 'Unknown')}: {e}. Skipping entry.")
            except Exception as e:
                logger.error(f"Unexpected error for unique_id {data['original_item'].get('unique_id', 'Unknown')}: {e}. Skipping entry.")

    logger.info(f"Parsing complete. Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Parse JSONL responses and generate CSV")
    parser.add_argument("--model", required=True, help="Model name to process responses for")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--include-text", action="store_true", help="Include text-heavy fields in the output")
    parser.add_argument("--include-metadata", action="store_true", help="Include metadata fields in the output")
    args = parser.parse_args()

    config = load_config(args.config)
    input_file = Path(config['data']['output_directory']) / f"api_responses_{args.model}_healed.jsonl"
    output_file = Path(config['data']['output_directory']) / f"processed_results_{args.model}.csv"

    logger.info(f"Processing responses for model: {args.model}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Including text-heavy fields: {args.include_text}")
    logger.info(f"Including metadata fields: {args.include_metadata}")

    parse_jsonl(str(input_file), str(output_file), args.model, args.include_text, args.include_metadata)
    logger.info(f"Parsing complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
