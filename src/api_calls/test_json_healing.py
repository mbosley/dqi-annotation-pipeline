#!/usr/bin/env python3

#!/usr/bin/env python3
import asyncio
import json
import logging
from typing import Dict, Any
import argparse
import aiohttp
from heal_llm_output import load_config, get_api_client, heal_json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_tests(config: Dict[str, Any]):
    schema = {
        "type": "object",
        "properties": {
            "participation": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer", "minimum": 0, "maximum": 3},
                    "reasoning": {"type": "string"}
                },
                "required": ["score", "reasoning"]
            },
            "justification": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "object",
                        "properties": {
                            "score": {"type": "integer", "minimum": 0, "maximum": 3},
                            "reasoning": {"type": "string"}
                        },
                        "required": ["score", "reasoning"]
                    }
                },
                "required": ["level"]
            }
        },
        "required": ["participation", "justification"]
    }
    healer_model_config = config['json_schema']['healing']['healer_model']
    client = get_api_client(healer_model_config, config['api_settings']['rate_limit'], config['global']['dry_run'], config['model_metadata'])

    test_cases = [
        {
            "name": "Missing comma",
            "input": '''{
                "participation": {
                    "score": 2
                    "reasoning": "The speaker actively contributes to the discussion."
                },
                "justification": {
                    "level": {
                        "score": 1,
                        "reasoning": "The speaker provides some justification for their claims."
                    }
                }
            }''',
            "expected_success": True
        },
        {
            "name": "Incorrect data type",
            "input": '''{
                "participation": {
                    "score": "high",
                    "reasoning": "The speaker is very engaged in the discussion."
                },
                "justification": {
                    "level": {
                        "score": 1,
                        "reasoning": "The speaker provides some justification for their claims."
                    }
                }
            }''',
            "expected_success": True
        },
        {
            "name": "Missing closing brace",
            "input": '''{
                "participation": {
                    "score": 2,
                    "reasoning": "The speaker actively contributes to the discussion."
                },
                "justification": {
                    "level": {
                        "score": 1,
                        "reasoning": "The speaker provides some justification for their claims."
                    }
            ''',
            "expected_success": True
        }
    ]

    async with aiohttp.ClientSession() as session:
        for test_case in test_cases:
            logger.info(f"Running test: {test_case['name']}")
            logger.info(f"Input JSON:\n{test_case['input']}")
            result = await heal_json(client, session, test_case['input'], schema, config['json_schema']['healing']['max_attempts'])

            if result.success == test_case['expected_success']:
                logger.info(f"Test '{test_case['name']}' passed.")
                if result.success:
                    logger.info(f"Healed JSON:\n{json.dumps(result.healed_json, indent=2)}")
            else:
                logger.error(f"Test '{test_case['name']}' failed. Expected success: {test_case['expected_success']}, Got: {result.success}")
                if result.error_message:
                    logger.error(f"Error message: {result.error_message}")
            logger.info(f"Healing attempts: {result.healing_attempts}")
            logger.info("---")

def main():
    parser = argparse.ArgumentParser(description="Run tests for JSON healing")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    asyncio.run(run_tests(config))

if __name__ == "__main__":
    main()
