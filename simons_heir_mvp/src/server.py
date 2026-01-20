import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .simulation import Simulation
from .predictor import Predictor
from .visualizer import plot_results
from .kalshi import KalshiClient
from .agent_generator import AgentGenerator
from .llm_interface import LlamaInterface
from .config import (
    SIMULATION_LOG_FILE,
    SENTIMENT_PLOT_FILE,
    AGENTS_COUNT,
    SIMULATION_DAYS,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simons_heir_server")

# Global State
class SimulationState:
    def __init__(self):
        self.simulation: Optional[Simulation] = None
        self.thread: Optional[threading.Thread] = None
        self.is_running: bool = False
        self.error: Optional[str] = None
        self.latest_analysis: Optional[pd.DataFrame] = None
    
    def reset(self):
        self.simulation = None
        self.thread = None
        self.is_running = False
        self.error = None
        self.latest_analysis = None

state = SimulationState()

# Models
class SimulationConfig(BaseModel):
    agents: int = AGENTS_COUNT
    days: int = SIMULATION_DAYS
    mock_mode: bool = True
    custom_agents: Optional[List[Dict[str, Any]]] = None

class SimulationStatus(BaseModel):
    is_running: bool
    current_step: int
    total_steps: int
    progress_pct: float
    current_price: float
    current_day: int
    run_error: Optional[str] = None
    recent_logs: List[Dict[str, Any]] = []

# Background Worker
def simulation_worker(config: SimulationConfig):
    """Running in a background thread"""
    global state
    try:
        logger.info("Starting simulation worker...")
        state.is_running = True
        state.error = None
        
        sim = Simulation(
            days=config.days,
            agent_count=config.agents,
            mock_llm=config.mock_mode,
            custom_agents=config.custom_agents
        )
        state.simulation = sim
        
        # This blocks until finished
        sim.run()
        
        # After run analysis
        logger.info("Simulation finished. Running analysis...")
        if not sim.stop_requested:
            predictor = Predictor()
            # Reload from file or use memory if we refactor predictor to accept list
            # The predictor.analyze() method accepts a list.
            state.latest_analysis = predictor.analyze(simulation_log=sim.simulation_log)
            
            # Generate plot immediately
            plot_results(state.latest_analysis, output_path=SENTIMENT_PLOT_FILE)
            
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        state.error = str(e)
    finally:
        state.is_running = False
        logger.info("Simulation worker finished")

# API
app = FastAPI(title="Simons' Heir API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class KalshiAnalysisResponse(BaseModel):
    trends: dict[str, Any]
    agents: list[dict[str, Any]]

class KalshiAgentsRequest(BaseModel):
    event_ticker: str
    count: int = 5

class KalshiAgentsResponse(BaseModel):
    event_ticker: str
    event_title: Optional[str] = None
    summary: Optional[str] = None
    agents: list[dict[str, Any]]

@app.post("/api/kalshi/analyze", response_model=KalshiAnalysisResponse)
async def analyze_kalshi_markets():
    try:
        # 1. Fetch trending events
        kalshi = KalshiClient()
        events = kalshi.get_trending_events(limit=20)
        
        # 2. Analyze trends
        analysis = kalshi.analyze_trends(events)
        
        return {
            "trends": analysis,
            "agents": []
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/kalshi/agents", response_model=KalshiAgentsResponse)
async def generate_kalshi_agents(payload: KalshiAgentsRequest):
    try:
        kalshi = KalshiClient()
        event = kalshi.get_event_details(payload.event_ticker)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        summary = kalshi.summarize_event(event)
        llm = LlamaInterface()
        generator = AgentGenerator(llm)
        agents = generator.generate_agents(summary, count=payload.count)

        return {
            "event_ticker": payload.event_ticker,
            "event_title": event.get("title"),
            "summary": summary,
            "agents": agents,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulation/start")
def start_simulation(config: SimulationConfig):
    if state.is_running:
        raise HTTPException(status_code=409, detail="Simulation already running")
    
    state.reset()
    state.thread = threading.Thread(target=simulation_worker, args=(config,)) # kept as tuple
    state.thread.start()
    
    return {"message": "Simulation started"}

@app.post("/api/simulation/stop")
def stop_simulation():
    if not state.is_running or not state.simulation:
        return {"message": "No simulation running"}
    
    state.simulation.stop_requested = True
    return {"message": "Stop signal sent"}

@app.get("/api/simulation/state", response_model=SimulationStatus)
def get_state():
    if not state.simulation:
        return SimulationStatus(
            is_running=False,
            current_step=0,
            total_steps=0,
            progress_pct=0.0,
            current_price=0.0,
            current_day=0,
            run_error=state.error
        )
    
    # Calculate progress
    # Simulation doesn't expose 'current_step' directly as a public counter easily
    # but we can infer from logs or time. 
    # Actually, let's fetch log length.
    steps_done = len(state.simulation.simulation_log)
    # Note: This is an approximation if multiple actions per step.
    # Ideally Simulation class should expose `current_step`.
    # Let's rely on time or just approximation for MVP.
    
    # Let's look at the implementation: 1 step = multiple agent actions.
    # The loop runs 'total_steps'. 
    # We need to expose current step in simulation.
    
    # For now, let's assume I'll add 'current_step_index' to Simulation class 
    # to make this accurate. If not, return 0.
    
    current_step = getattr(state.simulation, "_current_step_index", 0)
    total_steps = state.simulation.days * (24 // 1) # hardcoded config
    
    pct = (current_step / total_steps) * 100 if total_steps > 0 else 0
    
    # Get recent logs (last 5)
    recent = state.simulation.simulation_log[-5:] if state.simulation.simulation_log else []
    
    return SimulationStatus(
        is_running=state.is_running,
        current_step=current_step,
        total_steps=total_steps,
        progress_pct=round(pct, 1),
        current_price=state.simulation._current_price,
        current_day=(current_step // 24) + 1,
        run_error=state.error,
        recent_logs=recent
    )

@app.get("/api/results/stats")
def get_results_stats():
    if state.latest_analysis is None:
        # Try to load from disk if available
        if SIMULATION_LOG_FILE.exists():
            try:
                p = Predictor()
                state.latest_analysis = p.analyze(log_file=SIMULATION_LOG_FILE)
            except Exception:
                raise HTTPException(status_code=404, detail="No results available")
        else:
            raise HTTPException(status_code=404, detail="No results available")
            
    p = Predictor()
    summary = p.get_summary_stats(state.latest_analysis)
    
    # Convert dates to string for JSON serialization
    # Pandas timestamps need handling
    
    # Convert the whole DF to records for charting
    # Replace NaN with null
    chart_data = state.latest_analysis.where(pd.notnull(state.latest_analysis), None).to_dict(orient="records")
    
    return {
        "summary": summary,
        "chart_data": chart_data
    }

@app.get("/api/results/plot")
def get_results_plot():
    if not SENTIMENT_PLOT_FILE.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(SENTIMENT_PLOT_FILE)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
