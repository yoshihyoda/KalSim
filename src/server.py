import logging
import re
import threading
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .agent_generator import AgentGenerator
from .config import (
    AGENTS_COUNT,
    SENTIMENT_PLOT_FILE,
    SIMULATION_DAYS,
    SIMULATION_LOG_FILE,
    STEPS_PER_DAY,
)
from .kalshi import KalshiClient
from .llm_interface import LlamaInterface
from .predictor import Predictor
from .simulation import Simulation
from .visualizer import plot_results

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logger = logging.getLogger("kalsim_server")
compliance_logger = logging.getLogger("kalsim.compliance")


class SimulationState:
    def __init__(self) -> None:
        self.simulation: Simulation | None = None
        self.thread: threading.Thread | None = None
        self.is_running: bool = False
        self.error: str | None = None
        self.latest_analysis: pd.DataFrame | None = None

    def reset(self) -> None:
        self.simulation = None
        self.thread = None
        self.is_running = False
        self.error = None
        self.latest_analysis = None


state = SimulationState()
state_lock = threading.Lock()


class AgentBeliefs(BaseModel):
    risk_tolerance: str | None = None
    view: str | None = None
    market_outlook: str | None = None
    trust_in_institutions: str | None = None


class AgentSocial(BaseModel):
    follower_count: int | None = None
    influence_score: float | None = None


class AgentPersona(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int | None = None
    name: str | None = None
    personality_traits: list[str] | None = None
    interests: list[str] | None = None
    beliefs: AgentBeliefs | None = None
    social: AgentSocial | None = None


class SimulationLogEntry(BaseModel):
    agent_id: int
    timestamp: str
    action_type: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SimulationConfig(BaseModel):
    agents: int = AGENTS_COUNT
    days: int = SIMULATION_DAYS
    mock_mode: bool = False
    use_kalshi: bool = True
    custom_agents: list[AgentPersona] | None = None
    market_topic: str | None = None
    random_seed: int | None = None


class SimulationStatus(BaseModel):
    is_running: bool
    current_step: int
    total_steps: int
    progress_pct: float
    current_price: float
    current_day: int
    run_error: str | None = None
    recent_logs: list[SimulationLogEntry] = Field(default_factory=list)


class KalshiAnalysisResponse(BaseModel):
    trends: dict[str, Any]
    agents: list[AgentPersona]


class KalshiAgentsRequest(BaseModel):
    event_ticker: str
    count: int = Field(default=5, ge=1, le=100)


class KalshiAgentsResponse(BaseModel):
    event_ticker: str
    event_title: str | None = None
    summary: str | None = None
    agents: list[AgentPersona]


RESTRICTED_AGENT_CONTENT_FIELDS = {
    "text",
    "tweet",
    "tweets",
    "post",
    "posts",
    "body",
    "timeline",
    "message",
    "messages",
}


def _is_restricted_agent_field(field_name: str) -> bool:
    """Return True when field name suggests raw social content."""
    # Normalize snake_case/kebab-case/camelCase to token-like segments.
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", field_name)
    tokens = [tok for tok in normalized.replace("-", "_").lower().split("_") if tok]
    return any(token in RESTRICTED_AGENT_CONTENT_FIELDS for token in tokens)


def _normalize_agent_payload(agent: dict[str, Any], index: int) -> dict[str, Any]:
    """Normalize external agent payload fields before Pydantic validation."""
    normalized = dict(agent)
    name = normalized.get("name")

    if isinstance(name, str):
        cleaned_name = name.strip()
        normalized["name"] = cleaned_name if cleaned_name else f"Agent_{index}"
    elif name is None:
        normalized["name"] = f"Agent_{index}"
    else:
        normalized["name"] = str(name)

    dropped_fields = [
        key for key in list(normalized.keys()) if _is_restricted_agent_field(key)
    ]
    for key in dropped_fields:
        normalized.pop(key, None)

    if dropped_fields:
        compliance_logger.warning(
            "Dropped restricted content fields from agent payload: %s",
            ", ".join(sorted(dropped_fields)),
        )

    return normalized


def _validate_agent_personas(raw_agents: list[dict[str, Any]]) -> list[AgentPersona]:
    """Validate and coerce agent payloads into API response models."""
    return [
        AgentPersona.model_validate(_normalize_agent_payload(agent, idx))
        for idx, agent in enumerate(raw_agents)
    ]


def simulation_worker(config: SimulationConfig) -> None:
    """Run simulation in a background thread."""
    try:
        logger.info("Starting simulation worker...")
        with state_lock:
            state.is_running = True
            state.error = None

        sim = Simulation(
            days=config.days,
            agent_count=config.agents,
            mock_llm=config.mock_mode,
            custom_agents=[agent.model_dump(exclude_none=True) for agent in config.custom_agents]
            if config.custom_agents
            else None,
            use_kalshi=config.use_kalshi or not config.mock_mode,
            market_topic=config.market_topic,
            random_seed=config.random_seed,
            output_log_file=SIMULATION_LOG_FILE,
        )

        with state_lock:
            state.simulation = sim

        sim.run()

        logger.info("Simulation finished. Running analysis...")
        if not sim.stop_requested:
            predictor = Predictor()
            analysis_df = predictor.analyze(simulation_log=sim.simulation_log)
            plot_results(analysis_df, output_path=SENTIMENT_PLOT_FILE)
            with state_lock:
                state.latest_analysis = analysis_df

    except Exception as exc:
        logger.exception("Simulation failed: %s", exc)
        with state_lock:
            state.error = str(exc)
    finally:
        with state_lock:
            state.is_running = False
        logger.info("Simulation worker finished")


app = FastAPI(title="KalSim API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/kalshi/analyze", response_model=KalshiAnalysisResponse)
async def analyze_kalshi_markets() -> KalshiAnalysisResponse:
    try:
        kalshi = KalshiClient()
        events = kalshi.get_trending_events(limit=20)
        analysis = kalshi.analyze_trends(events)
        return KalshiAnalysisResponse(trends=analysis, agents=[])
    except Exception as exc:
        logger.exception("Kalshi analyze failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/kalshi/agents", response_model=KalshiAgentsResponse)
async def generate_kalshi_agents(payload: KalshiAgentsRequest) -> KalshiAgentsResponse:
    try:
        kalshi = KalshiClient()
        event = kalshi.get_event_details(payload.event_ticker)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        summary = kalshi.summarize_event(event)

        agents: list[dict[str, Any]] = []
        try:
            from .socioverse_connector import SocioVerseConnector

            connector = SocioVerseConnector()
            agents = connector.fetch_user_pool(count=payload.count)
            if agents:
                logger.info(
                    "Loaded %s personas from SocioVerse for market: %s",
                    len(agents),
                    payload.event_ticker,
                )
        except Exception as exc:
            logger.warning("SocioVerse fetch failed: %s", exc)

        if not agents:
            llm = LlamaInterface()
            generator = AgentGenerator(llm)
            agents = generator.generate_agents(summary, count=payload.count)
            logger.info(
                "Generated %s personas via LLM for market: %s",
                len(agents),
                payload.event_ticker,
            )

        return KalshiAgentsResponse(
            event_ticker=payload.event_ticker,
            event_title=event.get("title"),
            summary=summary,
            agents=_validate_agent_personas(agents),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Kalshi agent generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/simulation/start")
def start_simulation(config: SimulationConfig) -> dict[str, str]:
    with state_lock:
        if state.is_running:
            raise HTTPException(status_code=409, detail="Simulation already running")
        state.reset()
        state.thread = threading.Thread(target=simulation_worker, args=(config,), daemon=True)
        state.thread.start()

    return {"message": "Simulation started"}


@app.post("/api/simulation/stop")
def stop_simulation() -> dict[str, str]:
    with state_lock:
        sim = state.simulation
        is_running = state.is_running

    if not is_running or not sim:
        return {"message": "No simulation running"}

    sim.stop_requested = True
    return {"message": "Stop signal sent"}


@app.get("/api/simulation/state", response_model=SimulationStatus)
def get_state() -> SimulationStatus:
    with state_lock:
        sim = state.simulation
        is_running = state.is_running
        run_error = state.error

    if sim is None:
        return SimulationStatus(
            is_running=False,
            current_step=0,
            total_steps=0,
            progress_pct=0.0,
            current_price=0.0,
            current_day=0,
            run_error=run_error,
            recent_logs=[],
        )

    current_step = getattr(sim, "_current_step_index", 0)
    total_steps = sim.days * STEPS_PER_DAY
    pct = (current_step / total_steps) * 100 if total_steps > 0 else 0.0
    recent = sim.simulation_log[-5:] if sim.simulation_log else []

    return SimulationStatus(
        is_running=is_running,
        current_step=current_step,
        total_steps=total_steps,
        progress_pct=round(pct, 1),
        current_price=sim._current_price,
        current_day=(current_step // STEPS_PER_DAY) + 1 if total_steps > 0 else 0,
        run_error=run_error,
        recent_logs=[SimulationLogEntry.model_validate(log) for log in recent],
    )


@app.get("/api/results/stats")
def get_results_stats() -> dict[str, Any]:
    with state_lock:
        analysis_df = state.latest_analysis

    if analysis_df is None:
        if not SIMULATION_LOG_FILE.exists():
            raise HTTPException(status_code=404, detail="No results available")
        try:
            predictor = Predictor()
            analysis_df = predictor.analyze(log_file=SIMULATION_LOG_FILE)
            with state_lock:
                state.latest_analysis = analysis_df
        except Exception as exc:
            raise HTTPException(status_code=404, detail="No results available") from exc

    predictor = Predictor()
    summary = predictor.get_summary_stats(analysis_df)
    chart_data = analysis_df.where(pd.notnull(analysis_df), None).to_dict(orient="records")

    return {
        "summary": summary,
        "chart_data": chart_data,
    }


@app.get("/api/results/plot")
def get_results_plot() -> FileResponse:
    if not SENTIMENT_PLOT_FILE.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(SENTIMENT_PLOT_FILE)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
