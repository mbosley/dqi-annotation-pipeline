# Configuration
CONFIG ?= configs/polmeth-2024.yaml
CONFIG_NAME := $(notdir $(basename $(CONFIG)))
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
SRC_DIR := src
DATA_DIR := data/$(CONFIG_NAME)
CLEAN_DATA_DIR := $(DATA_DIR)/clean
PROMPTS_DATA_DIR := $(DATA_DIR)/prompts
API_DATA_DIR := $(DATA_DIR)/api_responses
RESULTS_DIR := results/$(CONFIG_NAME)
FIGS_DIR := $(RESULTS_DIR)/figures
BUILD_STAMPS_DIR := .build_stamps/$(CONFIG_NAME)

# Load model names from config
MODELS := $(shell $(PYTHON) -c "import yaml; print(' '.join(m['name'] for m in yaml.safe_load(open('$(CONFIG)'))['models']))")

# Stamp files
PREPARED_DATA_STAMP := $(BUILD_STAMPS_DIR)/prepared_data
PROMPT_STAMPS := $(patsubst %,$(BUILD_STAMPS_DIR)/prompts_%,$(MODELS))
API_CALL_STAMPS := $(patsubst %,$(BUILD_STAMPS_DIR)/api_calls_%,$(MODELS))
PARSE_STAMPS := $(patsubst %,$(BUILD_STAMPS_DIR)/parsed_%,$(MODELS))
EVALUATION_STAMP := $(BUILD_STAMPS_DIR)/evaluation

# Phony targets
.PHONY: all clean setup data prompts api_calls parse evaluate $(MODELS) help

# Default target
all: evaluate

# Create directories
$(BUILD_STAMPS_DIR) $(CLEAN_DATA_DIR) $(API_DATA_DIR) $(PROMPTS_DATA_DIR) $(RESULTS_DIR) $(FIGS_DIR):
	mkdir -p $@

# Create virtual environment and install dependencies
$(VENV)/bin/activate: requirements.txt | $(VENV)
	$(PIP) install -r $<
	@touch $@

$(VENV):
	python3 -m venv $@

# Setup: just an alias for creating the venv
setup: $(VENV)/bin/activate

# Data preparation
$(PREPARED_DATA_STAMP): $(VENV)/bin/activate | $(BUILD_STAMPS_DIR) $(CLEAN_DATA_DIR)
	@echo "Preparing data..."
	$(PYTHON) $(SRC_DIR)/clean_data/clean_label_data.py --config $(CONFIG)
	$(PYTHON) $(SRC_DIR)/clean_data/merge_us_debate_data.py --config $(CONFIG)
	@touch $@

data: $(PREPARED_DATA_STAMP)

# Generate prompts for all models
$(BUILD_STAMPS_DIR)/prompts_%: $(PREPARED_DATA_STAMP) | $(PROMPTS_DATA_DIR)
	@echo "Generating prompts for $*..."
	$(PYTHON) $(SRC_DIR)/clean_data/clean_data_for_prompt.py --model $* --config $(CONFIG)
	$(PYTHON) $(SRC_DIR)/prompt_generation/generate_prompts.py --model $* --config $(CONFIG)
	@touch $@

prompts: $(PROMPT_STAMPS)

# API calls for all models
$(BUILD_STAMPS_DIR)/api_calls_%: $(BUILD_STAMPS_DIR)/prompts_% | $(API_DATA_DIR)
	@echo "Processing API calls for $*..."
	$(PYTHON) $(SRC_DIR)/api_calls/send_prompts_to_api.py --model $* --config $(CONFIG)
	@touch $@

api_calls: $(API_CALL_STAMPS)

# Parse results for all models
$(BUILD_STAMPS_DIR)/parsed_%: $(BUILD_STAMPS_DIR)/api_calls_% | $(RESULTS_DIR)
	@echo "Parsing results for $*..."
	$(PYTHON) $(SRC_DIR)/api_calls/heal_llm_output_2.py --model $* --config $(CONFIG)
	$(PYTHON) $(SRC_DIR)/post_processing/parse_llm_responses_2.py --model $* --config $(CONFIG) --include-metadata
	@touch $@

parse: $(PARSE_STAMPS)

# Evaluate results
$(EVALUATION_STAMP): $(PARSE_STAMPS) | $(FIGS_DIR)
	@echo "Evaluating results..."
	$(PYTHON) $(SRC_DIR)/post_processing/merge_and_validate_results.py --config $(CONFIG)
	$(PYTHON) $(SRC_DIR)/post_processing/evaluate_results.py --config $(CONFIG)
	$(PYTHON) $(SRC_DIR)/post_processing/visualize_results.py --config $(CONFIG)
	@touch $@

evaluate: $(EVALUATION_STAMP)

# Individual model targets
$(MODELS): %: $(BUILD_STAMPS_DIR)/parsed_%
	@echo "Pipeline complete for $@"

# Clean up
clean:
	@echo "WARNING: This will remove the virtual environment and all generated data and results."
	@echo "Are you sure you want to proceed? (y/N)"
	@read answer; \
	if [ "$$answer" = "y" ] || [ "$$answer" = "Y" ]; then \
		rm -rf $(VENV) $(DATA_DIR) $(RESULTS_DIR) $(BUILD_STAMPS_DIR); \
	else \
		echo "Clean operation aborted."; \
	fi

# Help
help:
	@echo "Available targets:"
	@echo "  all       : Run the entire pipeline for all models (default)"
	@echo "  setup     : Set up the Python environment"
	@echo "  data      : Prepare and clean data"
	@echo "  prompts   : Generate prompts for all models"
	@echo "  api_calls : Make API calls for all models"
	@echo "  parse     : Parse results for all models"
	@echo "  evaluate  : Evaluate and visualize results"
	@echo "  clean     : Remove generated files and virtual environment"
	@echo "  <model>   : Run the pipeline for a specific model"
	@echo "  help      : Display this help message"
