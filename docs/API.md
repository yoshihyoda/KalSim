# API Reference

Base URL: `http://localhost:8000`

## Endpoints

### POST `/api/kalshi/analyze`

Fetches trending events and returns trend analysis.

### POST `/api/kalshi/agents`

Request body:

```json
{
  "event_ticker": "KXEVENT-EXAMPLE",
  "count": 5
}
```

Returns personas generated for the selected event.

Constraints:

- `count` must be between `1` and `100`.
- SocioVerse-backed persona fetches are only attempted when `KALSIM_RESEARCH_MODE=true`.
- Payload normalization drops raw social content-like fields (for example, `tweet_text`, `posts`, `messages`).

### POST `/api/simulation/start`

Request body:

```json
{
  "agents": 100,
  "days": 7,
  "mock_mode": true,
  "use_kalshi": true,
  "market_topic": "US climate goals",
  "random_seed": 42
}
```

Starts background simulation.

### POST `/api/simulation/stop`

Signals running simulation to stop early.

### GET `/api/simulation/state`

Returns current run status, progress, latest price, and recent logs.

### GET `/api/results/stats`

Returns summary metrics and chart data for the latest analysis.

### GET `/api/results/plot`

Returns generated plot image file.

## OpenAPI Sync Workflow

To refresh `docs/openapi.json` from the running code:

```bash
python scripts/generate_openapi.py
```

This script imports the FastAPI app and writes OpenAPI JSON to `docs/openapi.json`.
Update this document if endpoint semantics change.
