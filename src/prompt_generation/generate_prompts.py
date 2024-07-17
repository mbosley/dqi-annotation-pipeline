#!/usr/bin/env python3
import pandas as pd
import yaml
import argparse
import random
from typing import Dict, Any, List
import json
import os
import math
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_file: str = 'config.yaml') -> Dict[str, Any]:
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def load_prompt_template(elements: list) -> str:
    template = ""
    for element in elements:
        with open(element, 'r') as file:
            template += file.read() + "\n"
    return template

def select_examples(train_data: pd.DataFrame, n: int, seed: int) -> List[Dict[str, Any]]:
    random.seed(seed)
    return random.sample(train_data.to_dict('records'), min(n, len(train_data)))

def format_example(example: Dict[str, Any]) -> str:
    def safe_int(value):
        if isinstance(value, (int, float)) and not math.isnan(value):
            return int(value)
        return None

    rationales = json.loads(example['annotation_rationales'])
    simplified_example = {
        "annotation": {
            "participation": {
                "reasoning": rationales.get('Participation', "No rationale provided"),
                "score": safe_int(example.get('Participation'))
            },
            "justification": {
                "level": {
                    "reasoning": rationales.get('Justification-level', "No rationale provided"),
                    "score": safe_int(example.get('Justification-level'))
                },
                "content": {
                    "reasoning": rationales.get('Justification-content', "No rationale provided"),
                    "score": safe_int(example.get('Justification-content'))
                }
            },
            "respect": {
                "groups": {
                    "reasoning": rationales.get('Respect-group', "No rationale provided"),
                    "score": safe_int(example.get('Respect-group'))
                },
                "demand": {
                    "description": "Main demand in the speech",
                    "reasoning": rationales.get('Respect-demand', "No rationale provided"),
                    "score": safe_int(example.get('Respect-demand'))
                },
                "counterargument": {
                    "description": "Main counterargument in the speech",
                    "reasoning": rationales.get('Respect-counterarg', "No rationale provided"),
                    "score": safe_int(example.get('Respect-counterarg'))
                }
            },
            "constructive_politics": {
                "reasoning": rationales.get('Constructive Politics', "No rationale provided"),
                "score": safe_int(example.get('Constructive Politics'))
            }
        }
    }

    json_str = json.dumps(simplified_example, indent=2)
    return f"Speech: {example['speeches']}\n\nAnnotation:\n```json\n{json_str}\n```"

def construct_prompt(speech: str, previous_speeches: Any, template: str, examples: List[str], model_config: Dict[str, Any]) -> str:
    examples_text = "\n\n".join(f"Example {i+1}:\n{example}" for i, example in enumerate(examples)) if examples else "No examples provided."

    if isinstance(previous_speeches, float) and math.isnan(previous_speeches):
        previous_speeches = "No previous speeches available."
    else:
        previous_speeches = str(previous_speeches)

    default_closing_instructions = (
        "Please provide your annotation in JSON format according to the schema provided in the [JSON SCHEMA] section above. "
        "Format your JSON output with markdown-style backticks, like this:\n"
        "```json\n{Your JSON here}\n```\n"
        "Be sure to include ALL of the relevant information from the schema provided. Make sure that in your reasoning responses you consider step by step each of the possible annotation categories given the evidence at hand. Respond with only the JSON output. DO NOT INCLUDE UNESCAPED DOUBLE QUOTATION MARKS IN TEXT ENTRIES, USE SINGLE QUOTATION MARKS INSTEAD."
    )

    closing_instructions_path = model_config.get('prompt', {}).get('closing_instructions')

    if closing_instructions_path:
        with open(closing_instructions_path, 'r') as file:
            closing_instructions = file.read()
    else:
        closing_instructions = default_closing_instructions


    full_prompt = (
        f"{template}\n\n"
        f"[EXAMPLES]\n{examples_text}\n\n"
        f"[PREVIOUS SPEECHES]\n{previous_speeches}\n\n"
        f"[SPEECH TO ANNOTATE]\n{speech}\n\n"
        f"{closing_instructions}"
    )

    return full_prompt


def generate_prompts_for_initialization(train_data: pd.DataFrame, test_data: pd.DataFrame, model_config: Dict[str, Any], init_num: int, global_seed: int) -> List[Dict[str, Any]]:
    prompt_template = load_prompt_template(model_config['prompt']['elements'])
    n_examples = model_config['prompt'].get('n_icl_examples', 0)

    # Use a combination of global seed and init_num for deterministic but different sampling
    init_seed = global_seed + init_num
    examples = select_examples(train_data, n_examples, init_seed)
    formatted_examples = []
    for example in examples:
        try:
            formatted_examples.append(format_example(example))
        except Exception as e:
            logging.warning(f"Failed to format example: {e}")
            continue

    prompts = []
    for _, row in test_data.iterrows():
        prompt = construct_prompt(
            speech=row['speeches'],
            previous_speeches=row['previous_speeches'],
            template=prompt_template,
            examples=formatted_examples,
            model_config=model_config
        )
        prompts.append({
            'unique_id': f"{row['unique_id']}_{init_num}",
            'prompt': prompt
        })

    return prompts

def main():
    parser = argparse.ArgumentParser(description="Generate prompts for a specific model")
    parser.add_argument("--model", required=True, help="Model name to generate prompts for")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = next((model for model in config['models'] if model['name'] == args.model), None)
    if not model_config:
        raise ValueError(f"Model {args.model} not found in config")

    global_seed = config['global']['random_seed']
    n_random_init = model_config['prompt'].get('n_random_init', 1)

    train_data_path = f"{config['data']['clean_data_directory']}/processed_train_data_{args.model}.csv"
    test_data_path = f"{config['data']['clean_data_directory']}/processed_test_data_{args.model}.csv"

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    prompts_directory = config['data']['prompts_directory']
    os.makedirs(prompts_directory, exist_ok=True)

    for init in range(n_random_init):
        prompts = generate_prompts_for_initialization(train_data, test_data, model_config, init, global_seed)
        prompts_filename = os.path.join(prompts_directory, f'annotation_prompts_{args.model}_init-{init}.jsonl')

        with open(prompts_filename, 'w') as f:
            for prompt in prompts:
                f.write(json.dumps(prompt) + '\n')

        print(f"Prompts for initialization {init} saved to {prompts_filename}")

if __name__ == "__main__":
    main()
