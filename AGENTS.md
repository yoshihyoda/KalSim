# AGENTS.md

This file is for coding agents working in this repo.
It summarizes how to build, lint, test, and follow local style conventions.

## Repository Map
- Backend: `src/` (Python package), tests in `tests/`
- Frontend: `frontend/` (Vite + React + TypeScript)
- Data files: `data/`, results written to `results/`

## Build, Lint, Test

### Backend (Python)
- Install deps: `pip install -r requirements.txt`
- Run CLI simulation: `python -m src.main`
- Run API server: `python -m src.server`
- Alternative dev server: `uvicorn src.server:app --reload`
- Tests (all): `pytest`
- Tests with coverage: `pytest --cov=src --cov-report=term-missing`
- Single test file: `pytest tests/test_agent.py`
- Single test by name: `pytest tests/test_agent.py -k "decision"`

Notes:
- No dedicated Python formatter or linter config was found.
- Follow existing formatting and type hints in the code.

### Frontend (Vite + React + TS)
- Install deps: `npm install`
- Dev server: `npm run dev`
- Build: `npm run build`
- Lint: `npm run lint`
- Preview build: `npm run preview`

Notes:
- ESLint config: `frontend/eslint.config.js` (flat config)
- TypeScript is strict: `frontend/tsconfig.app.json`
- No frontend tests configured yet.

## Code Style Guidelines

### Python (Backend)
- Imports
  - Use stdlib, third-party, then local imports.
  - Local imports use relative paths (e.g., `from .config import ...`).
- Formatting
  - 4-space indentation and 88-100 char lines are common in the code.
  - Docstrings are used for modules, classes, and public functions.
- Types
  - Prefer modern type syntax (`list[str]`, `dict[str, Any]`, `A | None`).
  - Use `typing.Final` for constants in `config.py`.
  - Favor `dataclasses` for simple data containers.
- Naming
  - Modules, functions, and variables use `snake_case`.
  - Classes use `PascalCase`.
  - Constants are `UPPER_SNAKE_CASE` and live in `config.py`.
- Error handling
  - Catch and log exceptions with context (`logger.exception` or `logger.error`).
  - Use `HTTPException` in FastAPI routes for API errors.
  - Prefer explicit fallback behavior instead of silent failures.
- Logging
  - Use module-level `logger = logging.getLogger(__name__)`.
  - Keep logs structured and informative (phase banners in `main.py`).
- Paths and files
  - Use `pathlib.Path` (see `config.py`).
  - Avoid hard-coded strings for repo paths; reuse constants.

### FastAPI Patterns
- Pydantic models define request/response schemas.
- Background work runs in a thread (`simulation_worker`).
- Use response models for typed API responses.
- Use `state` for shared server state and guard concurrent runs.

### Frontend (React + TypeScript)
- Imports
  - Prefer explicit type imports with `import type { ... }`.
  - Keep local imports grouped and sorted (components, then styles).
- Formatting
  - Use double quotes in TS/TSX and keep JSX readable.
  - Components are function-based with hooks.
- Types
  - Use interfaces for API shapes in `frontend/src/api.ts`.
  - Avoid `any` where possible; prefer explicit fields or generics.
  - TS strict mode is enabled; fix `noUnusedLocals`/`noUnusedParameters`.
- Naming
  - Component files use `PascalCase.tsx` and export named components.
  - React component props are typed inline or with interfaces.
- State and effects
  - Keep `useEffect` cleanup functions and handle async errors.
  - Avoid side effects in render; place them in hooks.
- Styling
  - Tailwind utility classes are used for layout and visuals.
  - Prefer className composition over inline styles.

## Tests and Fixtures
- Pytest is the test runner; tests are under `tests/`.
- `tests/conftest.py` defines shared fixtures.
- Integration tests exist; some rely on external services.

## Data and Outputs
- Inputs live in `data/`.
- Simulation output log: `results/simulation_log.json`.
- Plot output: `results/sentiment_trend.png`.

## Cursor/Copilot Rules
- No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` found.
