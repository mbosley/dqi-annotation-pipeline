#!/usr/bin/env python3
import os
import yaml
import jsonlines
import asyncio
import aiohttp
import logging
import argparse
from asyncio import Semaphore
from typing import Dict, Any, List
from tqdm import tqdm
from dotenv import load_dotenv

# Import the necessary classes and functions from api_clients.py
from api_clients import APIResponse, APIClient, get_api_client

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add a file handler for detailed logging
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/api_client.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

def load_existing_results(output_file: str) -> Dict[str, Any]:
    existing_results = {}
    if os.path.exists(output_file):
        with jsonlines.open(output_file, 'r') as reader:
            for item in reader:
                if 'unique_id' in item and 'error' not in item:
                    existing_results[item['unique_id']] = item
    return existing_results

def load_config(config_file: str = 'config.yaml') -> Dict[str, Any]:
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

async def process_item(semaphore: Semaphore, session: aiohttp.ClientSession, client: APIClient, item: Dict[str, str], preload_str: str, prepend_str: str) -> Dict[str, Any]:
    async with semaphore:
        try:
            api_response = await client.send_request(session, item['prompt'], preload_str, item['unique_id'])
            logger.info(f"Successfully processed item {item['unique_id']}")

            # prepend the response -> content -> text field with the specified prepend_str
            api_response.response['content'][0]['text'] = prepend_str + api_response.response['content'][0]['text']

            return {
                'unique_id': api_response.unique_id,
                'prompt': api_response.prompt,
                'response': api_response.response,
                'prompt_tokens': api_response.prompt_tokens,
                'completion_tokens': api_response.completion_tokens,
                'total_tokens': api_response.total_tokens,
                'latency': api_response.latency,
                'request_id': api_response.request_id,
                'model_version': api_response.model_version,
                'estimated_cost': api_response.estimated_cost,
                'rate_limit_remaining': api_response.rate_limit_remaining
            }
        except Exception as e:
            logger.error(f"Error processing item {item['unique_id']}: {str(e)}")
            return {
                'unique_id': item['unique_id'],
                'prompt': item['prompt'],
                'error': str(e)
            }

async def process_initialization(input_file: str, output_file: str, config: Dict[str, Any], model_name: str, client: APIClient):
    model_config = next((model for model in config['models'] if model['name'] == model_name), None)
    if not model_config:
        raise ValueError(f"Model {model_name} not found in config")
    preload_str = model_config['model_params'].get('preload_str', '')
    prepend_str = model_config['model_params'].get('prepend_str', '')

    # Load existing results
    existing_results = load_existing_results(output_file)

    async with aiohttp.ClientSession() as session:
        with jsonlines.open(input_file, 'r') as reader, jsonlines.open(output_file, 'a') as writer:
            prompts = list(reader)

            if config['global']['test']:
                prompts = prompts[:config['global']['test_size']]
                logger.info(f"Running in test mode with {config['global']['test_size']} prompts")

            # Filter out prompts that have already been processed
            pending_prompts = [prompt for prompt in prompts if prompt['unique_id'] not in existing_results]
            logger.info(f"Found {len(existing_results)} existing results. Processing {len(pending_prompts)} pending prompts.")

            semaphore = Semaphore(config['api_settings']['concurrency'])
            tasks = [process_item(semaphore, session, client, item, preload_str, prepend_str) for item in pending_prompts]

            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Processing prompts for {model_name}"):
                try:
                    result = await f
                    writer.write(result)
                except Exception as e:
                    logger.error(f"Error: {str(e)}")

async def process_prompts(config: Dict[str, Any], model_name: str) -> None:
    model_config = next((model for model in config['models'] if model['name'] == model_name), None)
    if not model_config:
        raise ValueError(f"Model {model_name} not found in config")

    rate_limit = config['api_settings']['rate_limit']
    model_metadata = config['model_metadata']
    client = get_api_client(model_config, rate_limit, config['global']['dry_run'], model_metadata)

    n_random_init = model_config['prompt'].get('n_random_init', 1)

    for init in range(n_random_init):
        input_file = f"{config['data']['prompts_directory']}/annotation_prompts_{model_name}_init-{init}.jsonl"
        output_file = f"{config['data']['api_response_directory']}/api_responses_{model_name}_init-{init}.jsonl"

        logger.info(f"Processing initialization {init} for model {model_name}")
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output file: {output_file}")

        await process_initialization(input_file, output_file, config, model_name, client)

def main():
    parser = argparse.ArgumentParser(description="Send prompts to API for a specific model")
    parser.add_argument("--model", required=True, help="Model name to use for API calls")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    asyncio.run(process_prompts(config, args.model))

if __name__ == "__main__":
    main()
