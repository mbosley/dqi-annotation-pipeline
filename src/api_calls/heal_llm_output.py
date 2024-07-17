#!/usr/bin/env python3
import json
import jsonlines
import argparse
import logging
import yaml
from typing import Dict, Any, Optional, Tuple, List
import os
import asyncio
import aiohttp
from dataclasses import dataclass
from jsonschema import validate, ValidationError
from pathlib import Path
import re
import traceback
from dotenv import load_dotenv

# Import from api_clients.py
from api_clients import APIClient, get_api_client, APIResponse

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Add a file handler for detailed logging
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
    healing_method: str = "rule_based"
    error_excerpt: Optional[str] = None

def is_valid_json(text: str) -> bool:
    """Check if the given text is valid JSON."""
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False

def trim_to_json(text: str) -> str:
    """Trim text to contain only the JSON part."""
    participation_index = text.find("participation")
    if participation_index == -1:
        return text

    start_index = text.rfind("{", 0, participation_index)
    if start_index == -1:
        return text

    stack = []
    for i in range(start_index, len(text)):
        if text[i] == "{":
            stack.append("{")
        elif text[i] == "}":
            if stack and stack[-1] == "{":
                stack.pop()
                if not stack:
                    return text[start_index:i+1]

    return text[start_index:]

def restructure_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Restructure the JSON to nest 'demands' and 'counterarguments' under 'respect' if they are top-level properties.
    """
    restructured_data = data.copy()

    # Check if 'respect' exists, if not, create it
    if 'respect' not in restructured_data:
        restructured_data['respect'] = {}

    # Move 'demands' under 'respect' if it's a top-level property
    if 'demands' in restructured_data and 'demands' not in restructured_data['respect']:
        restructured_data['respect']['demands'] = restructured_data.pop('demands')

    # Move 'counterarguments' under 'respect' if it's a top-level property
    if 'counterarguments' in restructured_data and 'counterarguments' not in restructured_data['respect']:
        restructured_data['respect']['counterarguments'] = restructured_data.pop('counterarguments')

    return restructured_data

def apply_healing_steps(text: str) -> str:
    """Apply healing steps to the text."""
    healing_steps = [
        lambda t: re.sub(r'(\w+):', r'"\1":', t),
        lambda t: re.sub(r',\s*}', '}', t),
        lambda t: re.sub(r',\s*\]', ']', t),
        lambda t: re.sub(r':\s*"?\s*(\d+\.?\d*)\s*"?', r': \1', t),
        lambda t: re.sub(r'\\([^\\])', r'\1', t),
        lambda t: re.sub(r'"evidence":\s*\[([^\]]*)\]', lambda m: f'"evidence": [{process_evidence(m.group(1))}]', t)
    ]

    for step in healing_steps:
        text = step(text)

    return text

def process_evidence(evidence_str):
    lines = evidence_str.strip().split('\n')
    processed_lines = []
    for line in lines:
        line = line.replace('"', '')
        line = f'"{line.strip()}",'
        processed_lines.append(line)

    processed_evidence = '\n\n'.join(processed_lines)
    processed_evidence = processed_evidence.rstrip(',')

    return processed_evidence

def extract_error_excerpt(json_string: str, error_position: int, context_chars: int = 1000) -> str:
    """Extract an excerpt of the JSON string around the error position."""
    start = max(0, error_position - context_chars)
    end = min(len(json_string), error_position + context_chars)
    excerpt = json_string[start:end]

    if start > 0:
        excerpt = "..." + excerpt
    if end < len(json_string):
        excerpt = excerpt + "..."

    return excerpt

def extract_and_heal_json(response: Dict[str, Any], schema: Dict[str, Any], client: APIClient) -> Tuple[bool, str, List[str], Optional[str]]:
    healing_log = []
    error_excerpt = None

    # Extract content from response using the appropriate client
    text = client.extract_content(response)
    healing_log.append("Extracted content from response")

    if is_valid_json(text):
        healing_log.append("Content is already valid JSON")
    else:
        text = trim_to_json(text)
        healing_log.append("Trimmed content to JSON part")

        text = apply_healing_steps(text)
        healing_log.append("Applied healing steps")

    # Apply restructuring
    try:
        json_data = json.loads(text)
        restructured_data = restructure_json(json_data)
        text = json.dumps(restructured_data)
        healing_log.append("Applied JSON restructuring")
    except json.JSONDecodeError:
        healing_log.append("Failed to apply JSON restructuring due to invalid JSON")

    try:
        parsed_json = json.loads(text)
        validate(instance=parsed_json, schema=schema)
        healing_log.append("JSON successfully validated against schema")
        return True, json.dumps(parsed_json, indent=2), healing_log, None
    except json.JSONDecodeError as e:
        healing_log.append(f"JSON decoding error: {str(e)}")
        error_excerpt = extract_error_excerpt(text, e.pos)
    except ValidationError as e:
        healing_log.append(f"Schema validation error: {str(e)}")

    healing_log.append("No valid JSON found after all extraction and healing attempts")
    return False, text, healing_log, error_excerpt

class JSONHealer:
    def __init__(self, schema: Dict[str, Any], client: Optional[APIClient], session: Optional[aiohttp.ClientSession], max_attempts: int = 3, use_llm_fallback: bool = False):
        self.schema = schema
        self.client = client
        self.session = session
        self.max_attempts = max_attempts
        self.use_llm_fallback = use_llm_fallback

    async def heal_json(self, original_response: Dict[str, Any]) -> HealingResult:
        is_valid, healed_json, healing_log, error_excerpt = extract_and_heal_json(original_response, self.schema, self.client)

        if not is_valid:
            # Apply the restructuring
            restructured_json = restructure_json(json.loads(healed_json))
            healed_json = json.dumps(restructured_json)

            # Validate again after restructuring
            is_valid, healed_json, restructure_healing_log, error_excerpt = extract_and_heal_json({"content": [{"text": healed_json}]}, self.schema, self.client)
            healing_log.extend(restructure_healing_log)

        if is_valid:
            return HealingResult(
                original_json=json.dumps(original_response),
                healed_json=json.loads(healed_json),
                healing_attempts=1,
                success=True,
                healing_log=healing_log,
                healing_method="rule_based_with_restructuring"
            )

        if self.use_llm_fallback:
            for attempt in range(self.max_attempts):
                try:
                    llm_healed_json = await self.heal_with_llm(healed_json, "\n".join(healing_log))
                    is_valid, healed_json, llm_healing_log, error_excerpt = extract_and_heal_json({"content": [{"text": llm_healed_json}]}, self.schema, self.client)
                    healing_log.extend(llm_healing_log)

                    if is_valid:
                        return HealingResult(
                            original_json=json.dumps(original_response),
                            healed_json=json.loads(healed_json),
                            healing_attempts=attempt + 2,
                            success=True,
                            healing_log=healing_log,
                            healing_method="llm_based"
                        )
                except Exception as llm_error:
                    healing_log.append(f"LLM-based healing failed: {str(llm_error)}")

        return HealingResult(
            original_json=json.dumps(original_response),
            healed_json=None,
            healing_attempts=self.max_attempts + 1,
            success=False,
            error_message=f"Failed to heal JSON after {self.max_attempts + 1} attempts.",
            healing_log=healing_log,
            healing_method="failed",
            error_excerpt=error_excerpt
        )

    async def heal_with_llm(self, original_json: str, error_log: str) -> str:
        prompt = f"""The following JSON is invalid or does not conform to the required schema:

{original_json}

Errors encountered:
{error_log}

Please provide a corrected version of the JSON that resolves the errors and conforms to the required schema.
Make minimal changes necessary to fix the JSON. Preserve the original content and structure as much as possible.

ONLY PROVIDE JSON!!! DO NOT PROVIDE ANY COMMENTS ABOUT YOUR UPDATE TO THE JSON
"""

        api_response = await self.client.send_request(
            self.session, prompt,
            "Here is your requested JSON:\n{",
            "healing_request"
        )
        return self.client.extract_content(api_response.response)

def normalize_response_structure(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the structure of the response to a consistent format.
    """
    normalized_item = item.copy()

    if 'response' not in normalized_item:
        normalized_item['response'] = {}

    if 'content' not in normalized_item['response']:
        if 'choices' in normalized_item['response']:
            content = normalized_item['response']['choices'][0]['message']['content']
            normalized_item['response']['content'] = [{'text': content}]
        else:
            normalized_item['response']['content'] = [{'text': str(normalized_item['response'])}]

    return normalized_item

async def process_jsonl(input_file: str, output_file: str, config: Dict[str, Any], use_llm_fallback: bool, client: APIClient):
    schema = load_schema(config)
    session = None
    stats = HealingStats()

    if use_llm_fallback:
        session = aiohttp.ClientSession()

    try:
        healer = JSONHealer(schema, client, session, max_attempts=config['json_schema']['healing'].get('max_attempts', 3), use_llm_fallback=use_llm_fallback)
        with jsonlines.open(input_file, 'r') as reader, jsonlines.open(output_file, 'w') as writer:
            for item_number, item in enumerate(reader, start=1):
                stats.total_processed += 1
                try:
                    normalized_item = normalize_response_structure(item)
                    logger.debug(f"Processing item {item_number}. Original response: {json.dumps(normalized_item['response'])[:100]}...")

                    healing_result = await healer.heal_json(normalized_item['response'])

                    output_item = {
                        'original_item': item,
                        'healed_json': healing_result.healed_json if healing_result.success else None,
                        'healing_success': healing_result.success,
                        'healing_attempts': healing_result.healing_attempts,
                        'healing_method': healing_result.healing_method,
                        'healing_log': healing_result.healing_log
                    }

                    if healing_result.success:
                        stats.successfully_healed += 1
                    else:
                        stats.failed_healing += 1
                        output_item['healing_error'] = healing_result.error_message
                        if healing_result.error_excerpt:
                            output_item['error_excerpt'] = healing_result.error_excerpt

                    writer.write(output_item)

                except Exception as e:
                    stats.failed_healing += 1
                    logger.error(f"Error processing item {item_number}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    logger.error(f"Problematic item: {json.dumps(item, indent=2)}")
                    output_item = {
                        'original_item': item,
                        'healed_json': None,
                        'healing_success': False,
                        'processing_error': str(e),
                        'traceback': traceback.format_exc()
                    }
                    writer.write(output_item)
    finally:
        if session:
            await session.close()

    return stats

def log_healing_report(stats: HealingStats, model: str):
    success_rate = (stats.successfully_healed / stats.total_processed) * 100 if stats.total_processed > 0 else 0
    report = f"""
Healing Report for model {model}:
Total items processed: {stats.total_processed}
Successfully healed: {stats.successfully_healed}
Failed healing: {stats.failed_healing}
Success rate: {success_rate:.2f}%
"""
    logger.info(report)

    # Also write the report to a file
    report_file = os.path.join('logs', f'healing_report_{model}.txt')
    with open(report_file, 'w') as f:
        f.write(report)

def load_config(config_file: str = 'config.yaml') -> Dict[str, Any]:
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def load_schema(config: Dict[str, Any]) -> Dict[str, Any]:
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
    parser.add_argument("--llm-healing", action="store_true", help="Use LLM as fallback for JSON healing")
    args = parser.parse_args()

    config = load_config(args.config)

    model_config = next((model for model in config['models'] if model['name'] == args.model), None)
    if not model_config:
        raise ValueError(f"Model {args.model} not found in config")

    client = get_api_client(model_config, config['api_settings']['rate_limit'], config['global']['dry_run'], config['model_metadata'])

    input_file = os.path.join(config['data']['output_directory'], f"api_responses_{args.model}.jsonl")
    output_file = os.path.join(config['data']['output_directory'], f"api_responses_{args.model}_healed.jsonl")

    logging.info(f"Processing responses for model: {args.model}")
    logging.info(f"Input file: {input_file}")
    logging.info(f"Output file: {output_file}")
    logging.info(f"Healing method: Rule-based with {'LLM fallback' if args.llm_healing else 'no LLM fallback'}")

    stats = asyncio.run(process_jsonl(input_file, output_file, config, args.llm_healing, client))
    logging.info(f"Healing complete. Results saved to {output_file}")

    log_healing_report(stats, args.model)

if __name__ == "__main__":
    main()
