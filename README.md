# 🗳️ dqi-annotation-pipeline

Reusable primitives for automating Discourse Quality Index (DQI) annotation workflows.

## Scope

- Schema-first request/response contracts for DQI annotations
- Provider-agnostic model-run interfaces
- Structured output repair + validation utilities
- Evaluation primitives (accuracy, ordinal error, disagreement)
- Synthetic end-to-end examples and offline smoke checks
- Minimal clean interface in `src/core/`

## Out of scope

- Real datasets, real transcripts, or real study annotations
- Internal prompt assets from private project workflows
- Private logs, result bundles, or conference drafting materials
- Secrets, credentials, and environment-specific operational configs

## Project docs

- Plan: `docs/harvest-plan.md`
- Next steps: `docs/next-steps.md`
- Boundary policy: `policy/ip-boundary.md`
- Privacy/data policy: `policy/data-and-privacy.md`
- Prompt/eval policy: `policy/prompts-and-evals.md`

## Project citation

```bibtex
@article{bosley2025towards,
  author = {Bosley, Mitchell},
  title = {Towards Qualitative Measurement at Scale: A Prompt-Engineering Framework for Large-Scale Analysis of Deliberative Quality in Parliamentary Debates},
  journal = {Journal of Political Institutions and Political Economy},
  year = {2025},
  volume = {6},
  number = {3-4},
  pages = {355--383},
  doi = {10.1561/113.00000128}
}
```

Bosley, M. (2025). _Towards Qualitative Measurement at Scale: A Prompt-Engineering Framework for Large-Scale Analysis of Deliberative Quality in Parliamentary Debates_. Journal of Political Institutions and Political Economy, 6(3-4), 355–383. https://doi.org/10.1561/113.00000128

## Near-term roadmap

1. Ship stable core schemas (`specs/jsonschema/`)
2. Add synthetic fixtures and deterministic run manifests
3. Implement minimal end-to-end CLI for synthetic data
4. Add integration tests with mocked LLM responses
5. Publish `v0.1.0` as first reusable baseline

## Quick start

```bash
./tools/ip-scan.sh
./tools/smoke.sh
python3 -m unittest tests.integration.test_synthetic_end_to_end -v
```

## License

MIT
