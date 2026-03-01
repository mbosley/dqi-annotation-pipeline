# Security Policy

## Reporting a vulnerability

Please open a private security advisory on GitHub for this repository.

If private advisory flow is unavailable, open an issue with minimal detail and request a private follow-up channel.

## Secret handling expectations

- Never commit API keys, tokens, or private keys.
- Keep local secrets in ignored environment files.
- Treat synthetic test fixtures as public; do not include real user/study data.

## Current security posture

- Leak scan script: `./tools/ip-scan.sh`
- Offline smoke checks: `./tools/smoke.sh`
