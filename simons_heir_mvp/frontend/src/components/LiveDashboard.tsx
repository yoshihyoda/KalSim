import React, { useEffect, useRef } from "react";
import type { SimulationStatus } from "../api";
import {
  Activity,
  DollarSign,
  Users,
  StopCircle,
  Terminal,
} from "lucide-react";
import classNames from "classnames";

interface LiveDashboardProps {
  status: SimulationStatus;
  onStop: () => void;
}

export const LiveDashboard: React.FC<LiveDashboardProps> = ({
  status,
  onStop,
}) => {
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [status.recent_logs]);

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gray-800/80 backdrop-blur-xl p-6 rounded-2xl border border-gray-700/50 shadow-xl group hover:border-emerald-500/30 transition-all duration-300">
          <div className="flex items-center gap-3 text-emerald-400 mb-3 bg-emerald-500/10 w-fit p-2 rounded-lg">
            <Activity size={20} />
            <span className="text-sm font-bold tracking-wide uppercase">
              Progress
            </span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-4xl font-extrabold text-white tracking-tight">
              {status.progress_pct.toFixed(0)}
              <span className="text-2xl text-gray-500">%</span>
            </span>
            <span className="text-sm font-medium text-gray-400 mb-1 ml-auto bg-gray-900/50 px-3 py-1 rounded-full border border-gray-700">
              Day {status.current_day}
            </span>
          </div>
          <div className="w-full bg-gray-700/50 h-3 rounded-full mt-4 overflow-hidden shadow-inner">
            <div
              className="bg-linear-to-r from-emerald-500 to-cyan-500 h-full transition-all duration-500 ease-out shadow-[0_0_10px_rgba(16,185,129,0.5)]"
              style={{ width: `${status.progress_pct}%` }}
            />
          </div>
        </div>

        <div className="bg-gray-800/80 backdrop-blur-xl p-6 rounded-2xl border border-gray-700/50 shadow-xl group hover:border-amber-500/30 transition-all duration-300">
          <div className="flex items-center gap-3 text-amber-400 mb-3 bg-amber-500/10 w-fit p-2 rounded-lg">
            <DollarSign size={20} />
            <span className="text-sm font-bold tracking-wide uppercase">
              Current Price
            </span>
          </div>
          <div className="flex items-baseline gap-2">
            <span
              className={classNames(
                "text-4xl font-extrabold tracking-tight transition-colors drop-shadow-sm",
                status.current_price >= 20
                  ? "text-emerald-400"
                  : "text-rose-400",
              )}
            >
              ${status.current_price.toFixed(2)}
            </span>
          </div>
          <div className="text-xs font-medium text-gray-500 mt-4 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse"></span>
            Live Market Data
          </div>
        </div>

        <div className="bg-gray-800/80 backdrop-blur-xl p-6 rounded-2xl border border-gray-700/50 shadow-xl flex flex-col justify-between group hover:border-purple-500/30 transition-all duration-300">
          <div>
            <div className="flex items-center gap-3 text-purple-400 mb-3 bg-purple-500/10 w-fit p-2 rounded-lg">
              <Users size={20} />
              <span className="text-sm font-bold tracking-wide uppercase">
                Status
              </span>
            </div>
            <div className="text-xl font-medium text-white mb-2">
              {status.is_running ? (
                <span className="flex items-center gap-3 text-emerald-400 bg-emerald-500/10 py-2 px-3 rounded-lg border border-emerald-500/20">
                  <span className="relative flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                  </span>
                  <span className="font-bold">Simulation Active</span>
                </span>
              ) : (
                <span className="flex items-center gap-2 text-gray-400 bg-gray-700/30 py-2 px-3 rounded-lg border border-gray-700/50">
                  <div className="w-3 h-3 bg-gray-500 rounded-full"></div>
                  Paused / Finished
                </span>
              )}
            </div>
          </div>
          <button
            onClick={onStop}
            className="mt-4 w-full bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 hover:text-rose-300 border border-rose-500/30 hover:border-rose-500/50 py-3 px-4 rounded-xl text-sm font-bold transition-all flex items-center justify-center gap-2 shadow-lg hover:shadow-rose-900/20 active:scale-95 duration-200"
          >
            <StopCircle size={18} />
            Stop Simulation
          </button>
        </div>
      </div>

      {/* Live Feed */}
      <div className="bg-gray-900/90 backdrop-blur-xl rounded-2xl border border-gray-800 shadow-2xl overflow-hidden flex flex-col h-[500px]">
        <div className="bg-gray-800/50 px-6 py-4 border-b border-gray-700/50 flex items-center gap-3">
          <div className="p-1.5 bg-gray-700 rounded-md ring-1 ring-gray-600">
            <Terminal size={16} className="text-gray-300" />
          </div>
          <h3 className="text-sm font-mono font-bold text-gray-300 tracking-wide uppercase">
            Live Action Feed
          </h3>
          <div className="ml-auto flex gap-1.5">
            <div className="w-3 h-3 rounded-full bg-red-500/20 border border-red-500/50"></div>
            <div className="w-3 h-3 rounded-full bg-amber-500/20 border border-amber-500/50"></div>
            <div className="w-3 h-3 rounded-full bg-emerald-500/20 border border-emerald-500/50"></div>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-6 space-y-3 font-mono text-sm bg-black/20">
          {status.recent_logs.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-gray-600 space-y-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-500"></div>
              <p className="font-medium">Waiting for agent activity...</p>
            </div>
          ) : (
            status.recent_logs.map((log, i) => (
              <div
                key={i}
                className="flex gap-4 animate-fade-in group hover:bg-white/5 p-2 -mx-2 rounded-lg transition-colors border border-transparent hover:border-white/5"
              >
                <span className="text-gray-600 shrink-0 select-none w-20 text-right text-xs pt-0.5">
                  {new Date(log.timestamp).toLocaleTimeString([], {
                    hour12: false,
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                  })}
                </span>
                <div className="flex-1 break-all">
                  <span className="text-emerald-400 font-bold mr-3 bg-emerald-500/10 px-1.5 py-0.5 rounded text-xs">
                    AGENT_{log.agent_id}
                  </span>
                  <span
                    className={classNames(
                      "text-xs font-bold uppercase tracking-wider mr-2",
                      log.action_type === "TWEET"
                        ? "text-cyan-400"
                        : "text-purple-400",
                    )}
                  >
                    [{log.action_type}]
                  </span>
                  <span className="text-gray-300">{log.content}</span>
                </div>
              </div>
            ))
          )}
          <div ref={logsEndRef} />
        </div>
      </div>
    </div>
  );
};
