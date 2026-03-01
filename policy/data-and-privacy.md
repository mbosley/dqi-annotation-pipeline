# Data and Privacy

## Data policy

This repository is synthetic-first.

Only synthetic, public-safe example data should be stored in version control.

## Never commit

- Real speech transcripts or deliberation records
- Human annotation exports tied to identifiable participants
- Message/email/chat exports
- Any derived data that can be traced to a private source corpus

## Operational guidance

- Keep real data in separate private stores.
- Validate new fixtures for synthetic provenance before merge.
- Prefer generated toy examples for docs and tests.
