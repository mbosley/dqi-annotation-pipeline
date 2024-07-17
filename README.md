# Automated Discourse Quality Index (DQI) Annotation Pipeline

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*A scalable framework for automating Discourse Quality Index (DQI) annotations using Large Language Models*

[Key Features](#key-features) • [Pipeline Structure](#pipeline-structure) • [Makefile Execution](#makefile-execution) • [Configuration](#configuration) • [Prompt Engineering](#prompt-engineering) • [Adapting the Pipeline](#adapting-the-pipeline) • [Results and Visualization](#results-and-visualization)

</div>

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Pipeline Structure](#pipeline-structure)
- [Makefile Execution](#makefile-execution)
- [Configuration](#configuration)
- [Installation](#installation)
- [Detailed Usage Guide](#detailed-usage-guide)
- [Prompt Engineering](#prompt-engineering)
- [Adapting the Pipeline](#adapting-the-pipeline)
- [Adding New API Providers](#adding-new-api-providers)
- [Environment Configuration](#environment-configuration)
- [Results and Visualization](#results-and-visualization)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Disclaimer

This codebase is still very much a work in progress! It works well for me, but I cannot yet guarantee that it will work perfectly for you if you clone it and try to run it as-is. In particular, the codebase assumes that you've already downloaded the US congress debate data (https://data.stanford.edu/congress_text) as well as the DQI annotations. If you're interested in developing this kind of codebase as a package for general use, please reach out to me at mitchellbosley@gmail.com

## Introduction

This repository contains the comprehensive codebase for the Automated DQI Annotation Pipeline, as presented at PolMeth 2024. The pipeline leverages state-of-the-art Large Language Models (LLMs) to automate the annotation of political speeches using the Discourse Quality Index (DQI) framework (Steenbergen et al., 2003).

This approach demonstrates the feasibility of using LLMs for complex content analysis tasks in political science, with the goal of dramatically increasing the scale at which researchers can apply frameworks like DQI to large corpora of political discourse.

## Key Features

- 🚀 **Multi-Model Support**: Seamlessly integrate with OpenAI, Anthropic, DeepSeek, Meta, and other LLM providers.
- 📊 **Flexible In-Context Learning**: Experiment with zero-shot, few-shot, and many-shot learning approaches.
- 🧠 **Advanced Prompting Strategies**: Implement chain-of-thought (CoT) reasoning and other prompting techniques.
- 📈 **Comprehensive Evaluation**: Automatically calculate accuracy, F1 score, Mean Absolute Error (MAE), and other metrics.
- 💰 **Cost Analysis**: Track and analyze API usage costs for informed decision-making.
- 📊 **Visualization Suite**: Generate publication-ready figures for performance comparisons and cost-benefit analysis.

## Pipeline Structure

The pipeline is structured into several interconnected modules, each responsible for a specific part of the annotation process:

1. **Data Preparation** (`src/clean_data/`)
   - `clean_label_data.py`: Cleans and preprocesses the raw DQI-labeled dataset.
   - `merge_us_debate_data.py`: Merges multiple sources of US Congressional debate data.
   - `clean_data_for_prompt.py`: Prepares cleaned data for prompt generation, including handling of previous speeches for context.

2. **Prompt Generation** (`src/prompt_generation/`)
   - `generate_prompts.py`: Creates prompts for LLMs based on configuration settings, implementing various in-context learning strategies.

3. **API Integration** (`src/api_calls/`)
   - `api_clients.py`: Contains client classes for different LLM providers (OpenAI, Anthropic, DeepSeek, etc.).
   - `send_prompts_to_api.py`: Manages the process of sending prompts to LLMs and handling responses.

4. **Result Processing** (`src/post_processing/`)
   - `heal_llm_output_2.py`: Attempts to fix malformed JSON outputs from LLMs.
   - `parse_llm_responses_2.py`: Parses and structures the LLM responses for further analysis.

5. **Evaluation and Visualization** (`src/post_processing/`)
   - `evaluate_results.py`: Calculates performance metrics and conducts statistical tests.
   - `visualize_results.py`: Generates various plots and figures for analysis and presentation.

## Makefile Execution

The `Makefile` orchestrates the entire pipeline, allowing for both end-to-end execution and granular control over individual steps. Key targets include:

- `make setup`: Sets up the Python environment and installs dependencies.
- `make data`: Executes all data preparation steps.
- `make prompts`: Generates prompts for all configured models.
- `make api_calls`: Executes API calls for all models.
- `make parse`: Parses and processes the API responses.
- `make evaluate`: Evaluates results and generates visualizations.
- `make all`: Runs the entire pipeline for all configured models.

To run the pipeline for a specific model:

```bash
make build_MODEL_NAME
```

This command will execute all necessary steps for the specified model, from data preparation to evaluation.

## Configuration

The `configs/config.yaml` file is the central configuration point for the pipeline. It includes:

- **Global Settings**: Controls test mode, logging level, and other global parameters.
- **Data Settings**: Specifies input and output paths for various data files.
- **Model Configurations**: Defines settings for each LLM, including API details and prompting strategies.
- **In-Context Learning (ICL) Settings**: Controls the number of examples used for ICL and other related parameters.
- **API Settings**: Manages rate limiting and concurrency for API calls.
- **Evaluation Settings**: Specifies metrics to calculate and visualization options.

Example configuration snippet:

```yaml
models:
  - name: "gpt-4o"
    provider: "openai"
    api_key: "${OPENAI_API_KEY}"
    prompt:
      k: 4  # Number of previous speeches to include
      n: 2000  # Max length of concatenated previous speeches
      m: 500  # Max length of each previous speech
      p: 3000  # Max length of current speech
      include_previous_speeches: True
      n_icl_examples: 50
    model_params:
      model: "gpt-4o-2024-05-13"
      max_tokens: 4000
      temperature: 0.7
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/dqi-annotation-pipeline.git
   cd dqi-annotation-pipeline
   ```

2. Set up the environment:
   ```bash
   make setup
   ```

3. Configure API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Detailed Usage Guide

1. **Data Preparation**:
   - Place your raw data in the appropriate directory as specified in `config.yaml`.
   - Run `make data` to clean and preprocess the data.

2. **Prompt Generation**:
   - Adjust prompt templates in the `prompts/` directory if needed.
   - Run `make prompts` to generate prompts for all configured models.

3. **API Calls**:
   - Ensure your API keys are correctly set in the `.env` file.
   - Execute `make api_calls` to send prompts to the LLMs.

4. **Result Processing**:
   - Run `make parse` to process and structure the API responses.

5. **Evaluation and Visualization**:
   - Execute `make evaluate` to calculate metrics and generate visualizations.

6. **Full Pipeline Execution**:
   - Run `make all` to execute the entire pipeline for all configured models.

## Prompt Engineering

The effectiveness of LLMs in automating DQI annotations heavily relies on well-crafted prompts. Our pipeline implements advanced prompt engineering techniques to optimize model performance.

### Prompt Structure

Each prompt is structured into several key components:

1. **Task Description**: A clear explanation of the DQI annotation task and its importance.
2. **DQI Framework Overview**: A concise summary of the DQI dimensions and scoring criteria.
3. **In-Context Learning Examples**: Previously annotated speeches with explanations (number varies based on configuration).
4. **Current Speech**: The text to be annotated.
5. **Instruction for Output**: Specific guidelines on how to format the annotation response.

### Prompt Templates

Prompt templates are stored in the `prompts/` directory and are composed of modular components:

- `prompts/json_v2/opening-instructions.txt`: General task instructions
- `prompts/json_v2/dqi-theory.txt`: Theoretical background on DQI
- `prompts/json_v2/dqi-categories.txt`: Detailed explanation of DQI categories
- `prompts/json_v2/json-schema.txt`: Schema for structured output
- `prompts/json_v2/cot-closing-instructions.txt`: Instructions for chain-of-thought reasoning

### In-Context Learning (ICL)

The pipeline supports various ICL strategies:

- **Zero-shot**: No examples provided, relying solely on the model's pre-trained knowledge.
- **Few-shot**: A small number of examples (typically 1-5) to prime the model.
- **Many-shot**: Larger sets of examples (10-100+) to demonstrate patterns and edge cases.

Example configuration for ICL in `config.yaml`:

```yaml
prompt:
  n_icl_examples: 25  # Number of in-context learning examples
  n_random_init: 5    # Number of random initializations for example selection
```

### Chain-of-Thought (CoT) Reasoning

For complex annotations, the pipeline implements CoT prompting:

1. **Standard CoT**: Encourages the model to break down its reasoning process.
2. **Bayesian CoT**: Prompts the model to consider probabilities for each score.

### Customizing Prompts

To adapt the pipeline for new tasks:

1. Modify the template files in `prompts/` to reflect your annotation criteria.
2. Update the `prompt` section in `config.yaml` to use your new templates.
3. Adjust the `generate_prompts.py` script if your prompt structure differs significantly from the DQI format.

### Prompt Optimization Techniques

The pipeline includes several techniques for prompt optimization:

1. **Dynamic Example Selection**: Selects the most relevant in-context examples based on similarity to the target speech.
2. **Prompt Compression**: Efficiently compresses long contexts to fit within model token limits.
3. **Adaptive Prompting**: Adjusts prompt complexity based on the model's performance on previous samples.

### Experimenting with Prompt Variants

The pipeline supports easy experimentation with prompt variants by defining multiple prompt templates in the configuration file.

## Adapting the Pipeline

To adapt this pipeline for your own measurement task:

1. **Data Preparation**: Modify scripts in `src/clean_data/` to handle your specific dataset structure.
2. **Prompt Engineering**: Update templates in `prompts/` to reflect your annotation criteria and task-specific instructions.
3. **Evaluation Metrics**: Adjust `src/post_processing/evaluate_results.py` to include metrics relevant to your task.
4. **Configuration**: Modify `configs/config.yaml` to include your task-specific settings and model configurations.
5. **Visualization**: Extend `src/post_processing/visualize_results.py` to create task-specific plots and figures.

## Adding New API Providers

The pipeline is designed to be easily extensible to new LLM providers. To add a new API provider:

1. Create a new client class in `src/api_calls/api_clients.py`:

```python
class NewAPIClient(APIClient):
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _get_base_url(self) -> str:
        return self.model_config.get('base_url', 'https://api.newprovider.com/v1')

    def _prepare_request_data(self, prompt: str, preload: str) -> Dict[str, Any]:
        return {
            "model": self.model_config['model_params'].get('model', 'default-model'),
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": preload}
            ],
            "max_tokens": self.model_config['model_params'].get('max_tokens', 1024),
            "temperature": self.model_config['model_params'].get('temperature', 0.7),
        }

    def _get_api_endpoint(self) -> str:
        return "/completions"

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        # Implement response parsing logic
        pass

    def _extract_token_info(self, response: Dict[str, Any]) -> Dict[str, int]:
        # Implement token info extraction logic
        pass

    async def send_request(self, session: aiohttp.ClientSession, prompt: str, preload: str, unique_id: str) -> APIResponse:
        # Implement request sending logic
        pass

    def extract_content(self, response: Dict[str, Any]) -> str:
        # Implement content extraction logic
        pass
```

2. Update the `get_api_client` function in `src/api_calls/api_clients.py`:

```python
def get_api_client(model_config: Dict[str, Any], rate_limit: int, dry_run: bool, model_metadata: List[Dict[str, Any]]) -> APIClient:
    provider = model_config['provider']
    client_classes = {
        "anthropic": AnthropicClient,
        "deepseek": DeepSeekClient,
        "groq": GroqClient,
        "openai": OpenAIClient,
        "replicate": ReplicateClient,
        "together": TogetherClient,
        "newprovider": NewAPIClient  # Add your new API client here
    }
    client_class = client_classes.get(provider)
    if client_class:
        return client_class(model_config, rate_limit, dry_run, model_metadata)
    else:
        raise ValueError(f"Unsupported API provider: {provider}")
```

3. Add the new provider's configuration to `configs/config.yaml`:

```yaml
models:
  - name: "new-provider-model"
    provider: "newprovider"
    api_key: "${NEW_PROVIDER_API_KEY}"
    prompt:
      # ... [prompt configuration]
    model_params:
      model: "new-provider-model-name"
      max_tokens: 4000
      temperature: 0.7
```

4. Update the `model_metadata` section in `configs/config.yaml` to include pricing information for the new provider:

```yaml
model_metadata:
  - model: "new-provider-model-name"
    cost_per_million_input_tokens: 0.5
    cost_per_million_output_tokens: 1.5
    prompt_token_limit: 16000
    output_token_limit: 4000
```

## Environment Configuration

The pipeline uses environment variables to securely manage API keys and other sensitive information. To set up your environment:

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file with your actual API keys and other configuration values:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   REPLICATE_API_KEY=your_replicate_api_key_here
   TOGETHER_API_KEY=your_together_api_key_here
   NEW_PROVIDER_API_KEY=your_new_provider_api_key_here
   ```

3. Ensure that the `.env` file is listed in your `.gitignore` to prevent accidentally committing sensitive information.

4. The pipeline will automatically load these environment variables using the `python-dotenv` library.

To use these environment variables in your configuration, reference them in `configs/config.yaml` like this:

```yaml
models:
  - name: "gpt-4o"
    provider: "openai"
    api_key: "${OPENAI_API_KEY}"
    # ... rest of the configuration
```

The `${OPENAI_API_KEY}` syntax tells the configuration loader to substitute the value from the environment variable.

When adding a new API provider, make sure to update the `.env.example` file with the new environment variable, and document its usage in this README.

## Results and Visualization

The pipeline generates a comprehensive set of results and visualizations, including:

- Performance metrics (accuracy, F1 score, MAE) for each model and DQI dimension.
- Confusion matrices for each model's predictions.
- Plots comparing model performance across different numbers of ICL examples.
- Cost-benefit analysis visualizations.
- Dimension-wise heatmaps and radar charts.

All results are saved in the `results/` directory, with subdirectories for each model configuration.

Key visualizations include:

1. **Performance vs. ICL Examples**: Plots showing how model performance changes with the number of in-context learning examples.
2. **Model Comparison**: Bar charts and radar plots comparing different models across various metrics.
3. **Cost Analysis**: Scatter plots showing the trade-off between model performance and API costs.
4. **Confusion Matrices**: Heatmaps for each model and DQI dimension, showing classification performance.
5. **Dimension-wise Performance**: Heatmaps and radar charts showing model performance across different DQI dimensions.

To generate visualizations:

```bash
make evaluate
```

This command will process all results and create visualizations in the `results/figures/` directory.

## Performance Optimization

The pipeline includes several optimizations to improve efficiency and reduce API costs:

- **Concurrent API Calls**: The pipeline uses `asyncio` for efficient concurrent API requests, maximizing throughput within rate limits.
- **Caching**: Implemented caching mechanisms to avoid redundant API calls, particularly useful during development and debugging.
- **Batch Processing**: Option to process data in batches to manage memory usage for large datasets.
- **Smart Retrying**: Exponential backoff strategy for handling API rate limits and transient errors.

To adjust concurrency and rate limiting, modify the following settings in `configs/config.yaml`:

```yaml
api_settings:
  rate_limit: 10  # Requests per minute
  concurrency: 5  # Maximum number of concurrent requests
```

## Troubleshooting

Common issues and their solutions:

1. **API Rate Limiting**:
   - *Issue*: Receiving too many 429 (Too Many Requests) errors.
   - *Solution*: Adjust the `rate_limit` and `concurrency` settings in `config.yaml`. Decrease these values if you're hitting rate limits.

2. **JSON Parsing Errors**:
   - *Issue*: Failures in parsing LLM outputs.
   - *Solution*: Check the healing logs in `logs/healing_report_{model_name}.txt`. Adjust the JSON schema or healing logic in `src/api_calls/heal_llm_output_2.py` if necessary.

3. **Incorrect DQI Scores**:
   - *Issue*: LLM outputs don't match expected DQI scoring criteria.
   - *Solution*: Review and refine the prompt templates in `prompts/`. Consider adding more explicit instructions or examples.

4. **Visualization Errors**:
   - *Issue*: Failures in generating certain plots.
   - *Solution*: Ensure all required data is present in the results CSV files. Check for any data type mismatches in `src/post_processing/visualize_results.py`.

For more detailed error analysis, refer to the logs in the `logs/` directory. Each component of the pipeline generates its own log file, which can be invaluable for debugging.

## Contributing

We welcome contributions to improve and extend this pipeline! Here's how you can contribute:

1. **Reporting Issues**: If you encounter any bugs or have feature requests, please open an issue on the GitHub repository.

2. **Submitting Pull Requests**: 
   - Fork the repository and create your branch from `main`.
   - If you've added code, add tests that cover your new functionality.
   - Ensure your code adheres to the existing style. We use Black for Python code formatting.
   - Write a clear commit message describing your changes.
   - Open a pull request with a clear title and description.

3. **Improving Documentation**: Help us improve our documentation by submitting PRs for clarity, completeness, or to fix typos.

4. **Sharing Use Cases**: If you've used this pipeline for your research, we'd love to hear about it! Share your experience by opening an issue or contributing to the documentation.

Please see our [Contributing Guide](docs/contributing.md) for more detailed information on our development process, coding standards, and pull request procedure.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{bosley2024dqi_pipeline,
  author = {Bosley, Mitchell},
  title = {Automated Discourse Quality Index (DQI) Annotation Pipeline},
  year = {2024},
  url = {https://github.com/mcbosley/dqi-annotation-pipeline}
}
```

---

For questions, support, or collaboration opportunities, please open an issue on the GitHub repository or contact mitchellbosley@gmail.com.
