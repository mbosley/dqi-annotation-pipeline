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

## Foundational citation (DQI)

Steenbergen, M. R., Bächtiger, A., Spörndli, M., & Steiner, J. (2003). _Measuring Political Deliberation: A Discourse Quality Index_. Comparative European Politics, 1(1), 21–48. https://doi.org/10.1057/palgrave.cep.6110002

```bibtex
@article{steenbergen2003dqi,
  author = {Steenbergen, Marco R. and B{\"a}chtiger, Andr{\'e} and Sp{\"o}rndli, Markus and Steiner, J{\"u}rg},
  title = {Measuring Political Deliberation: A Discourse Quality Index},
  journal = {Comparative European Politics},
  year = {2003},
  volume = {1},
  number = {1},
  pages = {21--48},
  doi = {10.1057/palgrave.cep.6110002}
}
```

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
