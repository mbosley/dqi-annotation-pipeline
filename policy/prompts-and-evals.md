# Prompts and Evaluation Policy

## Prompt policy

Prompts in this repository must be generic and domain-agnostic.

Not allowed:
- Prompt packs from private research workflows
- Prompt text that includes private labels, entity names, or institution references

## Evaluation policy

- Use synthetic fixtures by default.
- Keep benchmark logic deterministic where feasible.
- Separate metric primitives (public) from study-specific benchmark corpora (private).

## Documentation requirement

When adding prompt or evaluation logic, document:
1. Input contract
2. Expected output schema
3. Known failure modes
