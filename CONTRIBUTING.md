# Contributing

Thanks for contributing.

## Scope boundary

This project accepts contributions for reusable, public-safe DQI primitives only.

Please do not submit:
- real datasets, transcripts, or annotation exports
- private prompt packs or client/study-specific templates
- secrets, credentials, or operational endpoint configs

## Development expectations

1. Keep interfaces deterministic and schema-driven.
2. Add tests for new behavior (unit first, then integration where relevant).
3. Use synthetic fixtures only.
4. Update docs when changing contracts.

## Pull request checklist

- [ ] No private/sensitive artifacts included
- [ ] `./tools/ip-scan.sh` passes
- [ ] `./tools/smoke.sh` passes
- [ ] Public docs updated for any interface changes
