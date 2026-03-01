# Next Steps

_Updated: March 1, 2026_

## Maintenance track (legacy + maintained)

1. Add mocked provider adapter tests for retry/error/timeout handling.
2. Add schema fixtures for common malformed output patterns.
3. Add a tiny `src/core/cli.py` for synthetic run execution and artifact writing.
4. Add CI to run `tools/smoke.sh` and integration tests on pull requests.
5. Remove or relocate legacy research artifacts that are outside public scope.

## Definition of done for current phase

- Core interface remains stable (`src/core/contracts.py` + `src/core/runner.py`).
- Synthetic end-to-end integration test remains green.
- Policy docs stay aligned with implementation scope.
