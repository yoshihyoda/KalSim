import React, { useState } from "react";
import { api } from "../api";
import type { SimulationConfig, KalshiAnalysisResponse } from "../api";

import {
  Settings,
  Play,
  Info,
  Loader2,
  TrendingUp,
  CheckCircle2,
  RefreshCw,
  Sparkles,
  Users,
  ExternalLink,
} from "lucide-react";

interface ConfigFormProps {
  onStart: (config: SimulationConfig) => void;
  isLoading: boolean;
}

export const ConfigForm: React.FC<ConfigFormProps> = ({
  onStart,
  isLoading,
}) => {
  const [config, setConfig] = useState<SimulationConfig>({
    agents: 100,
    days: 7,
    mock_mode: true,
  });
  const [analyzing, setAnalyzing] = useState(false);
  const [kalshiData, setKalshiData] = useState<KalshiAnalysisResponse | null>(
    null,
  );
  const [analysisTimestamp, setAnalysisTimestamp] = useState<Date | null>(null);
  const [showAllTopics, setShowAllTopics] = useState(false);
  const [showAllAgents, setShowAllAgents] = useState(false);
  const [selectedMarketIndex, setSelectedMarketIndex] = useState<number | null>(
    null,
  );
  const [selectedMarketTitle, setSelectedMarketTitle] = useState<string | null>(
    null,
  );
  const [generatingAgents, setGeneratingAgents] = useState(false);
  const [generatedAgents, setGeneratedAgents] = useState<any[]>([]);

  const handleAnalyze = async () => {
    setAnalyzing(true);
    try {
      const response = await api.analyzeKalshi();
      setKalshiData(response.data);
      setAnalysisTimestamp(new Date());
      setShowAllTopics(false);
      setShowAllAgents(false);
      setSelectedMarketIndex(null);
      setSelectedMarketTitle(null);
      setGeneratedAgents([]);
      setConfig((prev) => ({ ...prev, mock_mode: false }));
    } catch (error) {
      console.error("Failed to analyze markets:", error);
      alert("Failed to analyze Kalshi markets.");
    } finally {
      setAnalyzing(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onStart({
      ...config,
      custom_agents: generatedAgents.length > 0 ? generatedAgents : undefined,
    });
  };

  const topics = kalshiData?.trends?.topics ?? [];
  const tickers = kalshiData?.trends?.tickers ?? [];
  const eventTickers = kalshiData?.trends?.event_tickers ?? [];
  const seriesTickers = kalshiData?.trends?.series_tickers ?? [];
  const visibleTopics = showAllTopics ? topics : topics.slice(0, 8);
  const agents = generatedAgents;
  const visibleAgents = showAllAgents ? agents : agents.slice(0, 6);
  const marketBaseUrl = "https://kalshi.com/markets/";
  const selectedLabel =
    selectedMarketTitle ?? "Select a market to generate personas.";

  const handleSelectMarket = async (index: number) => {
    const eventTicker = eventTickers[index];
    if (!eventTicker) {
      return;
    }

    setSelectedMarketIndex(index);
    setSelectedMarketTitle(topics[index] ?? null);
    setGeneratingAgents(true);

    try {
      const response = await api.generateKalshiAgents(eventTicker);
      setGeneratedAgents(response.data.agents ?? []);
      setShowAllAgents(false);
    } catch (error) {
      console.error("Failed to generate agents:", error);
      alert("Failed to generate personas for the selected market.");
    } finally {
      setGeneratingAgents(false);
    }
  };

  return (
    <div className="w-full max-w-5xl mx-auto relative group transition-all duration-700 ease-out">
      {/* Abstract Background Glow */}
      <div className="absolute -inset-1 bg-linear-to-r from-emerald-500/20 via-cyan-500/20 to-blue-500/20 rounded-3xl blur-2xl opacity-50 group-hover:opacity-75 transition duration-1000"></div>

      <div className="relative bg-gray-900/80 backdrop-blur-2xl rounded-3xl border border-gray-800/60 shadow-2xl overflow-hidden">
        {/* Decorative top line */}
        <div className="absolute top-0 left-0 right-0 h-1 bg-linear-to-r from-transparent via-emerald-500/50 to-transparent opacity-50"></div>

        <div className="grid grid-cols-1 lg:grid-cols-12 divide-y lg:divide-y-0 lg:divide-x divide-gray-800/50">
          {/* Left Column: Configuration & Actions (4 cols) */}
          <div className="lg:col-span-4 p-8 flex flex-col justify-between bg-gray-900/50">
            <div className="space-y-8">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-gray-800/80 rounded-2xl border border-gray-700/50 shadow-inner">
                  <Settings className="text-emerald-400" size={24} />
                </div>
                <div>
                  <h2 className="text-lg font-bold text-white tracking-tight">
                    Simulation Setup
                  </h2>
                  <p className="text-[10px] text-gray-500 font-medium uppercase tracking-wider mt-0.5">
                    Configure Parameters
                  </p>
                </div>
              </div>

              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-4">
                  <div className="space-y-2 group/input">
                    <label className="text-xs font-semibold text-gray-400 uppercase tracking-widest ml-1 group-hover/input:text-emerald-400 transition-colors">
                      Agent Population
                    </label>
                    <div className="relative">
                      <input
                        type="number"
                        min="1"
                        max="1000"
                        value={config.agents}
                        onChange={(e) =>
                          setConfig({
                            ...config,
                            agents: parseInt(e.target.value) || 0,
                          })
                        }
                        className="w-full bg-gray-950/50 border border-gray-800 rounded-xl px-4 py-3.5 text-white font-mono focus:ring-1 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-all outline-none"
                      />
                      <div className="absolute right-4 top-1/2 -translate-y-1/2 text-xs text-gray-600 font-mono pointer-events-none">
                        AGT
                      </div>
                    </div>
                  </div>

                  <div className="space-y-2 group/input">
                    <label className="text-xs font-semibold text-gray-400 uppercase tracking-widest ml-1 group-hover/input:text-emerald-400 transition-colors">
                      Duration
                    </label>
                    <div className="relative">
                      <input
                        type="number"
                        min="1"
                        max="30"
                        value={config.days}
                        onChange={(e) =>
                          setConfig({
                            ...config,
                            days: parseInt(e.target.value) || 0,
                          })
                        }
                        className="w-full bg-gray-950/50 border border-gray-800 rounded-xl px-4 py-3.5 text-white font-mono focus:ring-1 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-all outline-none"
                      />
                      <div className="absolute right-4 top-1/2 -translate-y-1/2 text-xs text-gray-600 font-mono pointer-events-none">
                        DAYS
                      </div>
                    </div>
                  </div>
                </div>

                <div
                  className="flex items-center gap-3 p-3 rounded-xl border border-dashed border-gray-700 hover:border-emerald-500/30 hover:bg-emerald-500/5 transition-all cursor-pointer group/check"
                  onClick={() =>
                    setConfig((prev) => ({
                      ...prev,
                      mock_mode: !prev.mock_mode,
                    }))
                  }
                >
                  <div
                    className={`w-5 h-5 rounded-md flex items-center justify-center transition-all ${
                      config.mock_mode
                        ? "bg-emerald-500 text-white shadow-lg shadow-emerald-500/20"
                        : "bg-gray-800 text-gray-600 group-hover/check:bg-gray-700"
                    }`}
                  >
                    {config.mock_mode && (
                      <CheckCircle2 size={12} strokeWidth={3} />
                    )}
                  </div>
                  <div className="flex flex-col">
                    <span className="text-sm font-medium text-gray-300 group-hover/check:text-white transition-colors">
                      Mock LLM Mode
                    </span>
                    <span className="text-[10px] text-gray-500">
                      Simulated inference (Faster)
                    </span>
                  </div>
                </div>
              </form>
            </div>

            <div className="mt-8 space-y-3">
              <button
                type="button"
                onClick={handleAnalyze}
                disabled={analyzing || isLoading}
                className="w-full relative overflow-hidden group py-4 px-4 rounded-xl font-bold text-base transition-all border border-blue-500/30 hover:border-blue-400/50 bg-blue-500/5 hover:bg-blue-500/10 text-blue-400 cursor-pointer"
              >
                <div className="flex items-center justify-center gap-2 relative z-10">
                  {analyzing ? (
                    <Loader2 className="animate-spin" size={18} />
                  ) : (
                    <TrendingUp
                      size={18}
                      className="group-hover:scale-110 transition-transform"
                    />
                  )}
                  <span>
                    {analyzing
                      ? "Analyzing Market Data..."
                      : "Analyze Kalshi Markets"}
                  </span>
                </div>
              </button>

              <button
                onClick={handleSubmit}
                disabled={isLoading}
                className={`w-full relative overflow-hidden group py-4 px-4 rounded-xl font-bold text-base transition-all shadow-xl cursor-pointer ${
                  isLoading
                    ? "bg-gray-800 text-gray-500 cursor-not-allowed border border-gray-700"
                    : "bg-emerald-500 hover:bg-emerald-400 text-white shadow-emerald-500/20 hover:shadow-emerald-500/30 border border-emerald-400/20"
                }`}
              >
                <div className="absolute inset-0 bg-linear-to-r from-transparent via-white/10 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
                <div className="flex items-center justify-center gap-2 relative z-10">
                  {isLoading ? (
                    <>
                      <Loader2 className="animate-spin" size={20} />
                      <span>Initializing...</span>
                    </>
                  ) : (
                    <>
                      <Play size={20} fill="currentColor" />
                      <span>Start Simulation</span>
                    </>
                  )}
                </div>
              </button>
            </div>
          </div>

          {/* Right Column: Kalshi Results (8 cols) */}
          <div className="lg:col-span-8 bg-gray-900/20 min-h-[500px] flex flex-col">
            {!kalshiData ? (
              // Empty State
              <div className="h-full flex flex-col items-center justify-center p-12 text-center space-y-6">
                <div className="w-24 h-24 rounded-full bg-gray-800/50 border border-gray-700/50 flex items-center justify-center relative">
                  <div className="absolute inset-0 rounded-full border border-emerald-500/20 animate-ping opacity-20 duration-3000"></div>
                  <TrendingUp size={40} className="text-gray-600" />
                </div>
                <div>
                  <h3 className="text-lg font-medium text-gray-300">
                    Market Intelligence Inactive
                  </h3>
                  <p className="text-sm text-gray-500 max-w-xs mx-auto mt-2 leading-relaxed">
                    Connect to the Kalshi API to analyze real-time prediction
                    market trends and generate persona-based agents.
                  </p>
                </div>
              </div>
            ) : (
              // Results State
              <div className="p-8 h-full flex flex-col gap-6 animate-fade-in relative">
                {/* Header Strip */}
                <div className="flex items-center justify-between pb-6 border-b border-gray-800/50">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)] animate-pulse"></div>
                      <h3 className="text-sm font-bold text-emerald-400 uppercase tracking-widest">
                        Live Market Analysis
                      </h3>
                    </div>
                    <p className="text-xs text-gray-500 font-mono">
                      {selectedLabel} • {new Date().toLocaleTimeString()}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <span className="text-xs font-mono py-1 px-3 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-300">
                      {topics.length} MARKETS
                    </span>
                    <span className="text-xs font-mono py-1 px-3 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-300">
                      {agents.length} PERSONAS
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 grow">
                  {/* Topics Column */}
                  <div className="space-y-4">
                    <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-widest flex items-center gap-2">
                      <TrendingUp size={14} /> Trending Markets
                    </h4>
                    <div className="space-y-2">
                      {visibleTopics.map((topic, i) => {
                        const seriesTicker = seriesTickers[i];
                        const fallbackTicker = tickers[i] || eventTickers[i];
                        const marketTicker = seriesTicker || fallbackTicker;
                        const marketLink = marketTicker
                          ? `${marketBaseUrl}${encodeURIComponent(marketTicker)}`
                          : undefined;
                        const isSelected = selectedMarketIndex === i;

                        return (
                          <div
                            key={`${topic}-${i}`}
                            role="button"
                            tabIndex={0}
                            onClick={() => handleSelectMarket(i)}
                            onKeyDown={(event) => {
                              if (event.key === "Enter" || event.key === " ") {
                                event.preventDefault();
                                handleSelectMarket(i);
                              }
                            }}
                            className={`group flex items-start gap-3 p-3 rounded-xl border transition-all cursor-pointer ${
                              isSelected
                                ? "bg-blue-500/20 border-blue-500/50 shadow-[0_0_16px_rgba(59,130,246,0.25)]"
                                : "bg-gray-800/30 border-gray-700/30 hover:bg-gray-800/60 hover:border-blue-500/30"
                            }`}
                          >
                            <span className="shrink-0 flex items-center justify-center w-6 h-6 rounded-md bg-blue-500/10 text-[10px] font-bold text-blue-400 font-mono border border-blue-500/20">
                              {i + 1}
                            </span>
                            <div className="space-y-1 overflow-hidden flex-1">
                              <p className="text-xs font-medium text-gray-300 leading-snug group-hover:text-blue-200 transition-colors line-clamp-2">
                                {topic}
                              </p>
                              <p className="text-[10px] text-gray-600 font-mono tracking-wide truncate">
                                {eventTickers[i] || marketTicker || "N/A"}
                              </p>
                            </div>
                            {marketLink && (
                              <a
                                href={marketLink}
                                target="_blank"
                                rel="noopener noreferrer"
                                onClick={(event) => event.stopPropagation()}
                                className="flex items-center justify-center w-7 h-7 rounded-full border border-blue-500/20 bg-blue-500/10 text-blue-300 hover:text-blue-100 hover:border-blue-400/50 transition-colors"
                                aria-label="Open market on Kalshi"
                              >
                                <ExternalLink size={12} />
                              </a>
                            )}
                          </div>
                        );
                      })}
                    </div>
                    {topics.length > 8 && (
                      <button
                        onClick={() => setShowAllTopics(!showAllTopics)}
                        className="w-full text-center text-xs text-gray-500 hover:text-white py-2 transition-colors border-t border-gray-800/50 mt-2"
                      >
                        {showAllTopics
                          ? "Show Less"
                          : `View All ${topics.length} Markets`}
                      </button>
                    )}
                  </div>

                  {/* Agents Column */}
                  <div className="space-y-4">
                    <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-widest flex items-center gap-2">
                      <Users size={14} /> Generated Personas
                    </h4>
                    <div className="grid grid-cols-1 gap-3">
                      {generatingAgents && (
                        <div className="flex items-center justify-center gap-2 text-xs text-gray-400 py-6 border border-dashed border-gray-700/60 rounded-xl">
                          <Loader2 className="h-4 w-4 animate-spin" />
                          Generating personas for the selected market...
                        </div>
                      )}

                      {!generatingAgents && agents.length === 0 && (
                        <div className="text-xs text-gray-500 py-6 text-center border border-dashed border-gray-700/60 rounded-xl">
                          Select a market on the left to generate personas.
                        </div>
                      )}

                      {!generatingAgents &&
                        visibleAgents.map((agent: any, i: number) => {
                          const view = agent?.beliefs?.view || "Neutral";
                          const viewColor =
                            view === "Bullish"
                              ? "text-emerald-400"
                              : view === "Bearish"
                                ? "text-rose-400"
                                : "text-gray-400";

                          return (
                            <div
                              key={i}
                              className="p-3 rounded-xl bg-gray-800/30 hover:bg-gray-800/60 border border-gray-700/30 hover:border-emerald-500/30 transition-all group"
                            >
                              <div className="flex justify-between items-start mb-2">
                                <div className="flex items-center gap-2">
                                  <div className="w-6 h-6 rounded-full bg-linear-to-br from-gray-700 to-gray-800 flex items-center justify-center text-[10px] font-bold text-gray-300 border border-gray-600">
                                    {agent?.name
                                      ? agent.name[0].toUpperCase()
                                      : "?"}
                                  </div>
                                  <span className="text-xs font-bold text-white group-hover:text-emerald-300 transition-colors">
                                    @{agent?.name}
                                  </span>
                                </div>
                                <span
                                  className={`text-[10px] font-mono uppercase tracking-wider ${viewColor} bg-gray-900/50 px-2 py-0.5 rounded-md`}
                                >
                                  {view}
                                </span>
                              </div>
                              <div className="flex flex-wrap gap-1.5">
                                {(agent?.personality_traits || [])
                                  .slice(0, 3)
                                  .map((t: string, ti: number) => (
                                    <span
                                      key={ti}
                                      className="text-[10px] px-2 py-0.5 rounded-md bg-gray-900/40 border border-gray-700/50 text-gray-400"
                                    >
                                      {t}
                                    </span>
                                  ))}
                              </div>
                            </div>
                          );
                        })}
                    </div>
                    {agents.length > 6 && (
                      <button
                        onClick={() => setShowAllAgents(!showAllAgents)}
                        className="w-full text-center text-xs text-gray-500 hover:text-white py-2 transition-colors border-t border-gray-800/50 mt-2"
                      >
                        {showAllAgents
                          ? "Show Less"
                          : `View All ${agents.length} Agents`}
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Footer Info */}
      <div className="mt-6 flex justify-between items-center px-4 text-xs text-gray-500 font-mono">
        <div className="flex items-center gap-2">
          <Info size={14} className="text-emerald-500/50" />
          <span>Simons' Heir v1.0.0</span>
        </div>
        <div className="flex items-center gap-4">
          <span>READY</span>
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></div>
        </div>
      </div>
    </div>
  );
};
