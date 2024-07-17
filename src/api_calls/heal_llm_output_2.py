#!/usr/bin/env python3
import json
import jsonlines
import argparse
import logging
import yaml
from typing import Dict, Any, Optional, List
import os
from dataclasses import dataclass
from jsonschema import validate, ValidationError
from pathlib import Path
import re
from dotenv import load_dotenv

from api_clients import APIClient, get_api_client

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
file_handler = logging.FileHandler(log_dir / 'llm_parsing.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

@dataclass
class HealingStats:
    total_processed: int = 0
    successfully_healed: int = 0
    failed_healing: int = 0

@dataclass
class HealingResult:
    original_json: str
    healed_json: Optional[Dict[str, Any]]
    healing_attempts: int
    success: bool
    error_message: Optional[str] = None
    healing_log: List[str] = None

def is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False

def extract_json(text: str) -> str:
    """Extract JSON from text, with or without markdown-style backticks."""
    json_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    # If no backticks found, assume the entire text is JSON
    return text.strip()

def apply_healing_steps(text: str) -> str:
    """Apply healing steps to the text."""
    healing_steps = [
        # Add missing opening curly bracket if needed
        lambda t: '{' + t if t.strip().startswith('"') else t,
        # Existing steps
        lambda t: re.sub(r'(\w+):', r'"\1":', t),
        lambda t: re.sub(r',\s*}', '}', t),
        lambda t: re.sub(r',\s*\]', ']', t),
        lambda t: re.sub(r':\s*"?\s*(\d+\.?\d*)\s*"?', r': \1', t),
        lambda t: re.sub(r'\\([^\\])', r'\1', t),
        # Remove extra curly brackets
        lambda t: re.sub(r'{\s*{', '{', t),
        lambda t: re.sub(r'}\s*}', '}', t),
    ]

    for step in healing_steps:
        text = step(text)

    return text

def heal_json(text: str, schema: Optional[Dict[str, Any]]) -> HealingResult:
    healing_log = []

    # Extract JSON if it's surrounded by backticks
    text = extract_json(text)
    healing_log.append("Extracted JSON from markdown-style backticks")

    if is_valid_json(text):
        healing_log.append("Content is already valid JSON")
    else:
        text = apply_healing_steps(text)
        healing_log.append("Applied healing steps")

    try:
        parsed_json = json.loads(text)
        if schema:
            validate(instance=parsed_json, schema=schema)
            healing_log.append("JSON successfully validated against schema")
        else:
            healing_log.append("Schema validation skipped")
        return HealingResult(
            original_json=text,
            healed_json=parsed_json,
            healing_attempts=1,
            success=True,
            healing_log=healing_log
        )
    except json.JSONDecodeError as e:
        healing_log.append(f"JSON decoding error: {str(e)}")
    except ValidationError as e:
        healing_log.append(f"Schema validation error: {str(e)}")

    healing_log.append("Failed to heal JSON")
    return HealingResult(
        original_json=text,
        healed_json=None,
        healing_attempts=1,
        success=False,
        error_message="Failed to heal JSON",
        healing_log=healing_log
    )

def process_jsonl(input_file: str, output_file: str, config: Dict[str, Any], client: APIClient) -> HealingStats:
    schema = load_schema(config)
    stats = HealingStats()

    with jsonlines.open(input_file, 'r') as reader, jsonlines.open(output_file, 'w') as writer:
        for item_number, item in enumerate(reader, start=1):
            stats.total_processed += 1
            try:
                logger.debug(f"Processing item {item_number}. Original response: {json.dumps(item['response'])[:100]}...")

                content = client.extract_content(item['response'])
                healing_result = heal_json(content, schema)

                output_item = {
                    'original_item': item,
                    'healed_json': healing_result.healed_json,
                    'healing_success': healing_result.success,
                    'healing_log': healing_result.healing_log
                }

                if healing_result.success:
                    stats.successfully_healed += 1
                else:
                    stats.failed_healing += 1
                    output_item['healing_error'] = healing_result.error_message

                writer.write(output_item)

            except Exception as e:
                stats.failed_healing += 1
                logger.error(f"Error processing item {item_number}: {str(e)}")
                output_item = {
                    'original_item': item,
                    'healed_json': None,
                    'healing_success': False,
                    'processing_error': str(e)
                }
                writer.write(output_item)

    return stats

def log_healing_report(stats: HealingStats, model: str, init: int):
    success_rate = (stats.successfully_healed / stats.total_processed) * 100 if stats.total_processed > 0 else 0
    report = f"""
Healing Report for model {model}, initialization {init}:
Total items processed: {stats.total_processed}
Successfully healed: {stats.successfully_healed}
Failed healing: {stats.failed_healing}
Success rate: {success_rate:.2f}%
"""
    logger.info(report)

    report_file = os.path.join('logs', f'healing_report_{model}_init-{init}.txt')
    with open(report_file, 'w') as f:
        f.write(report)

def load_config(config_file: str = 'config.yaml') -> Dict[str, Any]:
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def load_schema(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not config['json_schema'].get('enabled', True):
        return None

    schema_path = config['json_schema']['path']
    try:
        with open(schema_path, 'r') as schema_file:
            return json.load(schema_file)
    except FileNotFoundError:
        logging.error(f"Schema file not found: {schema_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in schema file: {schema_path}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Heal malformed JSON in API responses")
    parser.add_argument("--model", required=True, help="Model name to process responses for")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    model_config = next((model for model in config['models'] if model['name'] == args.model), None)
    if not model_config:
        raise ValueError(f"Model {args.model} not found in config")

    client = get_api_client(model_config, config['api_settings']['rate_limit'], config['global']['dry_run'], config['model_metadata'])

    n_random_init = model_config['prompt'].get('n_random_init', 1)

    for init in range(n_random_init):
        input_file = os.path.join(config['data']['api_response_directory'], f"api_responses_{args.model}_init-{init}.jsonl")
        output_file = os.path.join(config['data']['api_response_directory'], f"api_responses_{args.model}_init-{init}_healed.jsonl")

        logging.info(f"Processing responses for model: {args.model}, initialization: {init}")
        logging.info(f"Input file: {input_file}")
        logging.info(f"Output file: {output_file}")

        stats = process_jsonl(input_file, output_file, config, client)
        logging.info(f"Healing complete for initialization {init}. Results saved to {output_file}")

        log_healing_report(stats, args.model, init)

if __name__ == "__main__":
    main()
