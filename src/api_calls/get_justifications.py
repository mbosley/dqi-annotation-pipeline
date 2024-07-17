#!/usr/bin/env python3
import pandas as pd
import argparse
import yaml
from typing import Dict, Any
from api_clients import get_api_client
import asyncio
import aiohttp
import json
import sys

def load_config(config_file: str = 'config.yaml') -> Dict[str, Any]:
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

async def generate_justification(client, session, row, config):
    prompt = f"""
Given the following speech and its DQI scores, generate a justification for each score:

Speech: {row['speech']}

Scores:
- Participation: {row['Participation']}
- Justification-level: {row['Justification-level']}
- Justification-content: {row['Justification-content']}
- Respect-group: {row['Respect-group']}
- Respect-demand: {row['Respect-demand']}
- Respect-counterarg: {row['Respect-counterarg']}
- Constructive Politics: {row['Constructive Politics']}

Provide a brief justification for each score, explaining why it was assigned based on the content of the speech.
"""
    justification_config = config['icl']['justification_model']
    response = await client.send_request(
        session,
        prompt,
        justification_config['model_params']['preload_str'],
        row['unique_id']
    )
    return json.loads(response.response['content'][0]['text'])

async def process_data(data: pd.DataFrame, config: Dict[str, Any]):
    justification_config = config['icl']['justification_model']
    client = get_api_client(justification_config, config['api_settings']['rate_limit'], config['global']['dry_run'], config['model_metadata'])

    async with aiohttp.ClientSession() as session:
        tasks = [generate_justification(client, session, row, config) for _, row in data.iterrows()]
        justifications = await asyncio.gather(*tasks)

    return justifications

def main():
    parser = argparse.ArgumentParser(description="Generate justifications for DQI scores")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    config = load_config(args.config)

    # Check if in-depth justifications are required
    if not config['icl']['use_in_depth_justifications']:
        print("In-depth justifications are not required. Exiting.")
        sys.exit(0)

    data = pd.read_csv(args.input)

    justifications = asyncio.run(process_data(data, config))

    with open(args.output, 'w') as f:
        json.dump(justifications, f, indent=2)

if __name__ == "__main__":
    main()
