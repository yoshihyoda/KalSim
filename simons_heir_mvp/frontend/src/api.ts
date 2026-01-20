import axios from "axios";

const API_BASE = "http://localhost:8000/api";

export interface SimulationConfig {
  agents: number;
  days: number;
  mock_mode: boolean;
  custom_agents?: any[];
}

export interface KalshiAnalysisResponse {
  trends: {
    topics: string[];
    summary: string;
    tickers?: string[];
    event_tickers?: string[];
    series_tickers?: string[];
  };
  agents: any[];
}

export interface KalshiAgentsResponse {
  event_ticker: string;
  event_title?: string;
  summary?: string;
  agents: any[];
}

export interface SimulationStatus {
  is_running: boolean;
  current_step: number;
  total_steps: number;
  progress_pct: number;
  current_price: number;
  current_day: number;
  run_error: string | null;
  recent_logs: any[];
}

export interface SimulationResults {
  summary: any;
  chart_data: any[];
}

export const api = {
  startSimulation: async (config: SimulationConfig) => {
    return axios.post(`${API_BASE}/simulation/start`, config);
  },

  stopSimulation: async () => {
    return axios.post(`${API_BASE}/simulation/stop`);
  },

  getState: async () => {
    return axios.get<SimulationStatus>(`${API_BASE}/simulation/state`);
  },

  getResultsStats: async () => {
    return axios.get<SimulationResults>(`${API_BASE}/results/stats`);
  },

  getResultsPlotUrl: () => {
    return `${API_BASE}/results/plot`;
  },

  analyzeKalshi: async () => {
    return axios.post<KalshiAnalysisResponse>(`${API_BASE}/kalshi/analyze`);
  },

  generateKalshiAgents: async (event_ticker: string, count = 5) => {
    return axios.post<KalshiAgentsResponse>(`${API_BASE}/kalshi/agents`, {
      event_ticker,
      count,
    });
  },
};
