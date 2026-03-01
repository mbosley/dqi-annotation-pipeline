# Congress Reports Harvest Inventory (Phase A)

_Updated: March 1, 2026_

This document inventories reusable, domain-agnostic primitives from `~/projects/congress-reports` for clean-room reimplementation in `delibtrace`.

## Candidate modules and interface specs

### 1) Workspace indexing and retrieval

- **Source inspiration:** `congress-reports/src/agentic/workspace.py`
- **Target in delibtrace:** `src/core/workspace_index.py`
- **Interface spec:**
  - Input: stage artifacts (`stage1`, `stage2`, `stage3`, optional `stage4`) as JSON dictionaries.
  - Output: deterministic indexes for `changes`, `semantic_changes`, `locations`, and optional `speech` references.
  - Required methods:
    - `get_change(change_id) -> dict | None`
    - `get_semantic_change(semantic_id) -> dict | None`
    - `list_semantic_changes() -> list[dict]`
    - `get_locations_for_semantic_change(semantic_id) -> list[str]`
  - Invariants:
    - stable ordering
    - no network calls
    - no mutation of input payloads

### 2) Provider client + mock client contract

- **Source inspiration:** `congress-reports/src/ai/client.py`
- **Target in delibtrace:** `src/core/providers.py`
- **Interface spec:**
  - `ProviderClient.analyze_json(prompt, system_message=None) -> dict`
  - `MockProviderClient(fixtures).analyze_json(...) -> dict`
  - Retry policy and timeout controls exposed via constructor config.
  - Invariants:
    - provider-agnostic interface
    - deterministic mock responses for tests
    - no provider-specific logic in downstream modules

### 3) Structured output validation layer

- **Source inspiration:** `congress-reports/src/ai/validators.py`
- **Target in delibtrace:** `src/core/validation.py`
- **Interface spec:**
  - `validate_annotation_result(payload: dict) -> dict`
  - `validate_relevance_assessment(payload: dict) -> dict`
  - `validate_influence_assessment(payload: dict) -> dict`
  - Shared error type: `ValidationError`
  - Invariants:
    - explicit required fields
    - explicit allowed categorical values
    - deterministic normalization rules

### 4) Text normalization and ID canonicalization

- **Source inspiration:** `congress-reports/src/utils/text_processing.py`, `src/utils/ids.py`
- **Target in delibtrace:** `src/core/text_utils.py`, `src/core/id_utils.py`
- **Interface spec:**
  - `clean_text(text: str) -> str`
  - `normalize_whitespace(text: str) -> str`
  - `parse_section_number(header: str) -> str | None`
  - `build_location_id(namespace, section, subsection=None, paragraph=None) -> str`
  - Invariants:
    - pure functions
    - no hidden global state
    - reversible ID structure where possible

### 5) Change analytics and summary statistics

- **Source inspiration:** `congress-reports/src/utils/diff_analysis.py`, `src/analysis/scoring.py`
- **Target in delibtrace:** `src/core/change_metrics.py`
- **Interface spec:**
  - `filter_changes_by_type(changes, change_type) -> list[dict]`
  - `group_changes_by_section(changes) -> dict[str, list[dict]]`
  - `summarize_change_set(changes) -> dict`
  - Optional scoring contract for qualitative dimensions mapped to numeric outputs.
  - Invariants:
    - deterministic summary math
    - explicit handling of missing fields
    - schema-compatible output payloads

### 6) Cost and token accounting

- **Source inspiration:** `congress-reports/src/utils/cost_tracker.py`
- **Target in delibtrace:** `src/core/costs.py`
- **Interface spec:**
  - `track_call(model, prompt_tokens, completion_tokens) -> None`
  - `get_summary() -> dict`
  - `format_summary() -> str`
  - Invariants:
    - configurable pricing table
    - resettable state
    - optional singleton wrapper for run-level accounting

## Exclusions for harvest

- Congressional or committee-specific prompt wording
- Domain-specific taxonomy labels tied to the private workflow
- Real run artifacts, transcripts, and speech-level outputs
- Project-specific stage scripts and local environment assumptions

## Implementation order recommendation

1. Workspace indexing + text/ID utilities
2. Validation layer + provider contract (with mocks)
3. Change metrics and cost tracking
4. Integrate into `src/core/cli.py` with synthetic fixtures
