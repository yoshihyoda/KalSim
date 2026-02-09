# Data Policy

## Bundled Data

Data under `data/` is intended for sample/demo usage and must be redistributable.

## Synthetic-Only Rule

For this OSS release, bundled datasets are treated as synthetic samples.
Do not add proprietary, personal, or restricted datasets directly into the repo.

## External Data Sources

- Kalshi: fetched at runtime via public APIs where applicable
- SocioVerse: accessed externally, subject to Hugging Face dataset terms

## SocioVerse Compliance Rules

- Research-only: SocioVerse use in this repository is restricted to research experiments.
- Explicit opt-in: Set `KALSIM_RESEARCH_MODE=true` to enable SocioVerse access.
- Terms acceptance: Contributors/operators must accept dataset conditions on Hugging Face before use.
- X content handling: For X-related data, text content must be obtained via the official X API.
- Metadata-only integration: This codebase only uses persona-like attributes and excludes raw social post text fields.
- Auditability: SocioVerse access events are logged through the `kalsim.compliance` logger.

Operational note:

- If research mode is disabled, SocioVerse fetches are blocked and the system falls back to non-SocioVerse persona generation paths.

## Contributor Requirements

When adding or updating bundled data:
- Document source and license status
- Confirm redistribution rights
- Remove personal/sensitive data
- Update `data/README.md`
