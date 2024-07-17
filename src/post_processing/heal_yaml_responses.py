#!/usr/bin/env python3
import yaml
import jsonlines
import asyncio
import aiohttp
import logging
from typing import Dict, Any, List
from ..api_calls.send_prompts_to_api import AnthropicClient, OpenAIClient, GroqClient, DeepSeekClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_file: str = 'config.yaml') -> Dict[str, Any]:
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

config = load_config()

def get_api_client(healer_config: Dict[str, Any], rate_limit: int, dry_run: bool):
    provider = healer_config['provider']
    if provider == 'anthropic':
        return AnthropicClient(healer_config, rate_limit, dry_run)
    elif provider == 'openai':
        return OpenAIClient(healer_config, rate_limit, dry_run)
    elif provider == 'groq':
        return GroqClient(healer_config, rate_limit, dry_run)
    elif provider == 'deepseek':
        return DeepSeekClient(healer_config, rate_limit, dry_run)
    else:
        raise ValueError(f"Unsupported API provider: {provider}")

async def heal_yaml(client, session: aiohttp.ClientSession, original_prompt: str, malformed_yaml: str, max_attempts: int = 3) -> str:
    attempt = 0
    while attempt < max_attempts:
        healing_prompt = f"""
The following YAML is malformed:

{malformed_yaml}

Please correct the YAML and ensure it is properly formatted. Return only the corrected YAML without any additional explanations or markdown formatting.
"""
        try:
            response = await client.send_request(session, healing_prompt)
            corrected_yaml = response['choices'][0]['message']['content'].strip()

            # Try to parse the corrected YAML
            yaml.safe_load(corrected_yaml)

            # If parsing succeeds, return the corrected YAML
            return corrected_yaml
        except yaml.YAMLError:
            logger.warning(f"Attempt {attempt + 1} failed to produce valid YAML. Retrying...")
            attempt += 1
        except Exception as e:
            logger.error(f"Error during YAML healing: {str(e)}")
            attempt += 1

    logger.error(f"Failed to heal YAML after {max_attempts} attempts.")
    return malformed_yaml  # Return the original malformed YAML if healing fails

async def process_responses(input_file: str, output_file: str, healer_config: Dict[str, Any]):
    client = get_api_client(healer_config, config['api_settings']['rate_limit'], config['global']['dry_run'])

    async with aiohttp.ClientSession() as session:
        with jsonlines.open(input_file, 'r') as reader, jsonlines.open(output_file, 'w') as writer:
            for item in reader:
                try:
                    # Try to parse the YAML content
                    yaml_content = item['response']['choices'][0]['message']['content']
                    yaml.safe_load(yaml_content)
                except yaml.YAMLError:
                    logger.warning(f"Malformed YAML detected for item {item['unique_id']}. Attempting to heal...")
                    healed_yaml = await heal_yaml(client, session, item['prompt'], yaml_content)
                    item['response']['choices'][0]['message']['content'] = healed_yaml
                    item['yaml_healed'] = True
                except KeyError:
                    logger.error(f"Unexpected response structure for item {item['unique_id']}")
                except Exception as e:
                    logger.error(f"Error processing item {item['unique_id']}: {str(e)}")

                writer.write(item)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Heal malformed YAML in API responses")
    parser.add_argument("--model", required=True, help="Model name to process responses for")
    parser.add_argument("--healer", default="gpt-4o", help="API provider to use for YAML healing")
    args = parser.parse_args()

    input_file = f"{config['data']['output_directory']}/api_responses_{args.model}.jsonl"
    output_file = f"{config['data']['output_directory']}/api_responses_{args.model}_healed.jsonl"

    # Get the healer configuration
    healer_config = next((model for model in config['models'] if model['name'] == args.healer), None)
    if not healer_config:
        raise ValueError(f"Healer model {args.healer} not found in config")

    asyncio.run(process_responses(input_file, output_file, healer_config))

if __name__ == "__main__":
    main()
