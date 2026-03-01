# Congress Reports Integration Plan

_Updated: March 1, 2026_

## Goal

Harvest reusable, public-safe components from `~/projects/congress-reports` into `delibtrace` to expand deliberative-quality workflows while preserving clean boundaries.

Phase A inventory artifact: `docs/congress-reports-harvest-inventory.md`.

## Integration targets

1. **Data normalization primitives**
   - Generic speech/document normalization utilities
   - Stable metadata schema for debate records
2. **Prompt assembly patterns**
   - Reusable prompt composition and templating interfaces
   - Provider-agnostic prompt packaging contracts
3. **Evaluation utilities**
   - Error bucketing and disagreement analysis helpers
   - Reporting structures for model-vs-reference comparisons
4. **Run artifact conventions**
   - Deterministic manifest/log layout for repeatable runs
   - Portable result bundles for downstream analysis

## Guardrails

- No direct copy of private/client-specific assets.
- Synthetic fixtures only in this public repo.
- Keep source provenance notes in private ops records, not in public code artifacts.

## Execution phases

### Phase A: Inventory

- Identify candidate modules in `congress-reports` that are domain-agnostic.
- Write concise interface specs for each candidate before implementation.
- Status: completed on March 1, 2026.

### Phase B: Clean-room reimplementation

- Rebuild selected components under `src/core/` or package-specific modules.
- Add schema and integration tests with synthetic data.

### Phase C: Validation and adoption

- Run smoke + integration tests in `delibtrace`.
- Update docs and migration notes for usage across deliberation projects.
