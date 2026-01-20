import { useState, useEffect } from "react";
import { api } from "./api";
import type { SimulationConfig, SimulationStatus } from "./api";
import { ConfigForm } from "./components/ConfigForm";
import { LiveDashboard } from "./components/LiveDashboard";
import { ResultsView } from "./components/ResultsView";
import { TrendingUp, LayoutDashboard, Database } from "lucide-react";

type AppMode = "CONFIG" | "RUNNING" | "RESULTS";

function App() {
  const [mode, setMode] = useState<AppMode>("CONFIG");
  const [status, setStatus] = useState<SimulationStatus>({
    is_running: false,
    current_step: 0,
    total_steps: 0,
    progress_pct: 0,
    current_price: 20.0,
    current_day: 1,
    run_error: null,
    recent_logs: [],
  });
  const [isLoading, setIsLoading] = useState(false);

  // Polling for status
  useEffect(() => {
    let interval: number;

    if (mode === "RUNNING") {
      interval = setInterval(async () => {
        try {
          const { data } = await api.getState();
          setStatus(data);

          if (!data.is_running && data.progress_pct >= 100) {
            setMode("RESULTS");
          }

          if (data.run_error) {
            alert(`Simulation failed: ${data.run_error}`);
            setMode("CONFIG");
          }
        } catch (error) {
          console.error("Poll failed", error);
        }
      }, 1000);
    }

    return () => clearInterval(interval);
  }, [mode]);

  const handleStart = async (config: SimulationConfig) => {
    setIsLoading(true);
    try {
      await api.startSimulation(config);
      setMode("RUNNING");
    } catch (error) {
      alert("Failed to start simulation. Ensure backend is running.");
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleStop = async () => {
    try {
      await api.stopSimulation();
      // Don't switch mode immediately, let poll detect stop
    } catch (error) {
      console.error(error);
    }
  };

  const handleRestart = () => {
    setMode("CONFIG");
    setStatus({ ...status, progress_pct: 0, recent_logs: [] });
  };

  return (
    <div className="min-h-screen bg-transparent text-gray-100 font-sans selection:bg-emerald-500/30">
      {/* Navbar */}
      <nav className="fixed top-0 w-full z-50 bg-gray-900/80 backdrop-blur-md border-b border-gray-800 px-6 py-4 flex items-center justify-between transition-all duration-300">
        <div className="flex items-center gap-3">
          <div className="bg-linear-to-tr from-emerald-500 to-cyan-500 p-2 rounded-lg">
            <TrendingUp size={24} className="text-white" />
          </div>
          <h3 className="text-xl font-bold tracking-tight text-white">
            Prediction Market Predictor
          </h3>
        </div>
        <div className="flex items-center gap-6 text-sm font-medium text-gray-400">
          <NavItem
            icon={<LayoutDashboard size={18} />}
            label="Dashboard"
            active={mode !== "RESULTS"}
          />
          <NavItem
            icon={<Database size={18} />}
            label="Analysis"
            active={mode === "RESULTS"}
          />
        </div>
      </nav>

      {/* Main Content */}
      <main className="container mx-auto px-6 pt-32 pb-12">
        {mode === "CONFIG" && (
          <div className="animate-fade-in mt-10">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-white mb-2">
                Configure Simulation
              </h2>
              <p className="text-sm text-gray-400">
                Set the parameters for the agent-based market simulation.
              </p>
            </div>
            <ConfigForm onStart={handleStart} isLoading={isLoading} />
          </div>
        )}

        {mode === "RUNNING" && (
          <div className="animate-fade-in">
            <LiveDashboard status={status} onStop={handleStop} />
          </div>
        )}

        {mode === "RESULTS" && (
          <div className="animate-fade-in">
            <ResultsView onRestart={handleRestart} />
          </div>
        )}
      </main>
    </div>
  );
}

const NavItem = ({
  icon,
  label,
  active,
}: {
  icon: React.ReactNode;
  label: string;
  active: boolean;
}) => (
  <div
    className={`flex items-center gap-2 transition-colors ${active ? "text-emerald-400" : "hover:text-white"}`}
  >
    {icon}
    <span>{label}</span>
  </div>
);

export default App;
