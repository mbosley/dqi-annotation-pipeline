# DQI Clean-Room Harvest Plan

## Goal

Harvest a public, reusable DQI annotation toolkit from prior private research work without copying private artifacts, datasets, or project-specific prompts.

## Source Context (private inspiration only)

Reference repo inspected: `/Users/mitchellbosley/Desktop/projects/dqi-annotation-pipeline`

The private repo currently mixes:
- reusable pipeline primitives (`src/`, `schemas/`)
- research artifacts (`paper/`, `jobtalk/`, poster PDFs)
- operational artifacts (`logs/`, `results/`, `data/`, `venv/`)
- prompt experiments (`prompts/`) tied to specific study workflows

## Clean-Room Rules

1. No direct copy/paste from private source files.
2. Re-implement from interface behavior/specification only.
3. Use synthetic fixtures and synthetic transcripts for tests/examples.
4. Keep adapters generic; no project/client nouns.

## Public Scope (to build here)

1. **Schema layer**
   - DQI request/response JSON schemas
   - validation + error reporting contracts
2. **Model runner layer**
   - provider-agnostic interface for LLM calls
   - retries, rate limits, and deterministic run manifests
3. **Output repair layer**
   - structured-output healing for malformed model JSON
   - explicit confidence + trace metadata
4. **Evaluation layer**
   - metric computation (classification + ordinal outcomes)
   - disagreement and error-bucket reporting
5. **CLI/API layer**
   - single entrypoint to run synthetic end-to-end demos

## Explicit Exclusions

- Real speeches, real annotations, raw datasets, or transformed real datasets
- Real logs/results/transcripts from prior runs
- Internal prompt wording from private workflows
- Any secrets, tokens, or environment-specific endpoints
- Paper drafting assets and conference materials

## Phase Plan

### Phase 0 — Governance and boundary hardening

- Add boundary docs (`policy/`) and leak scan tooling (`tools/ip-scan.sh`)
- Add contributing/security docs and baseline CI assumptions
- Exit criteria: boundary docs and scanners are in place

### Phase 1 — Spec-first baseline

- Publish core JSON schemas and interface contracts
- Add synthetic examples and schema validation smoke tests
- Exit criteria: schemas validate and examples pass offline

### Phase 2 — Minimal clean-room implementation

- Implement generic parser/runner/repair/eval primitives
- Add deterministic run artifact format
- Exit criteria: one synthetic end-to-end run path works locally

### Phase 3 — Stabilization and release

- Add test matrix (unit + integration with mocked LLM responses)
- Add docs and migration notes from private research workflows
- Exit criteria: tagged `v0.1.0` with reproducible synthetic demo

## Acceptance Criteria for “Shareable Repo”

- A new user can run the synthetic pipeline with no private data
- Repo contains no client/study-sensitive artifacts
- Public docs clearly define what is included and excluded
- Core interfaces are stable enough to reuse in other projects
