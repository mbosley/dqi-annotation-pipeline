# IP Boundary

## Principle

Public repository content must remain a clean-room implementation of reusable interfaces and behavior.

## Allowed

- Generic schema definitions
- Generic orchestration/validation logic
- Synthetic examples and fixtures
- Public documentation of methods

## Disallowed

- Direct transfer of private repository source files
- Client/study-specific prompts and templates
- Real logs, transcripts, annotations, and result exports
- Internal links, account details, and private infrastructure references

## Review gate

Every release should include:
1. Leak scan (`./tools/ip-scan.sh`)
2. Synthetic-only verification for examples/tests
3. README + policy review for scope drift
