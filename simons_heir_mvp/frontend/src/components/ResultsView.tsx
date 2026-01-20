import React, { useEffect, useState } from "react";
import { api } from "../api";
import { Line } from "react-chartjs-2";
import type { ChartData } from "chart.js";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Download, RefreshCw, Activity } from "lucide-react";

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
);

interface ResultsViewProps {
  onRestart: () => void;
}

export const ResultsView: React.FC<ResultsViewProps> = ({ onRestart }) => {
  const [chartData, setChartData] = useState<ChartData<"line"> | null>(null);
  const [summary, setSummary] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const { data } = await api.getResultsStats();
        setSummary(data.summary);

        // Process chart data
        const labels = data.chart_data.map(
          (row: any) =>
            new Date(row.timestamp).toLocaleDateString() +
            " " +
            new Date(row.timestamp).toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            }),
        );

        const sentiment = data.chart_data.map(
          (row: any) => row.sentiment_score,
        );
        const price = data.chart_data.map((row: any) => row.market_price);

        setChartData({
          labels,
          datasets: [
            {
              label: "Sentiment",
              data: sentiment,
              borderColor: "#34d399", // emerald-400
              backgroundColor: "rgba(52, 211, 153, 0.5)",
              yAxisID: "y",
            },
            {
              label: "Stock Price ($)",
              data: price,
              borderColor: "#fbbf24", // amber-400
              backgroundColor: "rgba(251, 191, 36, 0.5)",
              yAxisID: "y1",
            },
          ],
        });
      } catch (error) {
        console.error("Failed to load results", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="text-center text-gray-400 mt-20">
        Loading analysis results...
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-8 pb-12">
      <div className="flex justify-between items-center bg-gray-900/50 backdrop-blur-sm p-4 rounded-2xl border border-gray-800">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-linear-to-r from-emerald-500/20 to-cyan-500/20 rounded-xl border border-emerald-500/20">
            <Activity className="text-emerald-400" size={24} />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white tracking-tight">
              Simulation Analysis
            </h2>
            <p className="text-sm text-gray-400">
              Performance Metrics & Correlation Grid
            </p>
          </div>
        </div>

        <div className="flex gap-3">
          <button
            onClick={onRestart}
            className="flex items-center gap-2 px-5 py-2.5 bg-gray-800 hover:bg-gray-700/80 rounded-xl text-white border border-gray-700 hover:border-gray-600 transition-all shadow-lg active:scale-95"
          >
            <RefreshCw size={18} className="text-gray-400" />
            New Simulation
          </button>
          <a
            href={api.getResultsPlotUrl()}
            download="sentiment_plot.png"
            className="flex items-center gap-2 px-5 py-2.5 bg-linear-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 rounded-xl text-white transition-all shadow-lg shadow-emerald-900/30 hover:shadow-emerald-900/50 active:scale-95 border border-emerald-500/50"
          >
            <Download size={18} />
            Download Report
          </a>
        </div>
      </div>

      {/* Stats Cards */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          <StatCard
            label="Total Tweets Generated"
            value={summary.total_tweets}
            color="emerald"
          />
          <StatCard
            label="Average Sentiment"
            value={summary.avg_sentiment.toFixed(3)}
            color="cyan"
            subtext={
              summary.avg_sentiment > 0
                ? "Positive Outlook"
                : "Negative Outlook"
            }
          />
          <StatCard
            label="Peak Price Point"
            value="$N/A"
            subtext="(See Chart)"
            color="amber"
          />
          <StatCard
            label="Peak Activity Time"
            value={
              summary.peak_activity_hour?.split("T")[1]?.slice(0, 5) || "N/A"
            }
            color="purple"
          />
        </div>
      )}

      {/* Chart */}
      <div className="bg-gray-800/80 backdrop-blur-xl p-8 rounded-2xl border border-gray-700/50 shadow-2xl relative group">
        <div className="absolute inset-0 bg-linear-to-b from-gray-800/50 to-transparent rounded-2xl pointer-events-none"></div>
        <div className="h-[500px] relative z-10">
          {chartData && (
            <Line
              data={chartData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                  mode: "index",
                  intersect: false,
                },
                plugins: {
                  legend: {
                    position: "top",
                    labels: {
                      color: "#9ca3af",
                      font: { family: "Inter, sans-serif" },
                      usePointStyle: true,
                      pointStyle: "circle",
                    },
                  },
                  title: {
                    display: false,
                  },
                  tooltip: {
                    backgroundColor: "rgba(17, 24, 39, 0.9)",
                    titleColor: "#f3f4f6",
                    bodyColor: "#d1d5db",
                    borderColor: "rgba(75, 85, 99, 0.4)",
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                    displayColors: true,
                  },
                },
                scales: {
                  x: {
                    grid: { color: "rgba(55, 65, 81, 0.3)" },
                    ticks: { color: "#6b7280" },
                  },
                  y: {
                    type: "linear",
                    display: true,
                    position: "left",
                    grid: { color: "rgba(55, 65, 81, 0.3)" },
                    ticks: { color: "#6b7280" },
                    title: {
                      display: true,
                      text: "Sentiment Score",
                      color: "#34d399",
                      font: { weight: "bold" },
                    },
                  },
                  y1: {
                    type: "linear",
                    display: true,
                    position: "right",
                    grid: { drawOnChartArea: false },
                    ticks: { color: "#fbbf24" },
                    title: {
                      display: true,
                      text: "Market Price ($)",
                      color: "#fbbf24",
                      font: { weight: "bold" },
                    },
                  },
                },
              }}
            />
          )}
        </div>
      </div>

      {/* Keyword Stats */}
      {summary?.keyword_totals && (
        <div className="bg-gray-800/80 backdrop-blur-xl p-8 rounded-2xl border border-gray-700/50 shadow-2xl">
          <div className="flex items-center gap-3 mb-6">
            <div className="h-8 w-1 bg-emerald-500 rounded-full"></div>
            <h3 className="text-xl font-bold text-white tracking-tight">
              Trending Keywords
            </h3>
          </div>
          <div className="flex flex-wrap gap-4">
            {Object.entries(summary.keyword_totals)
              .sort(([, a], [, b]) => (b as number) - (a as number))
              .map(([key, val]) => (
                <div
                  key={key}
                  className="bg-gray-900/80 hover:bg-gray-800 px-5 py-2.5 rounded-xl border border-gray-700 hover:border-emerald-500/50 transition-all flex items-center gap-3 group cursor-default"
                >
                  <span className="text-emerald-400 font-medium group-hover:text-emerald-300 transition-colors">
                    #{key}
                  </span>
                  <span className="bg-gray-800 text-xs px-2.5 py-1 rounded-md text-gray-300 group-hover:bg-gray-700 group-hover:text-white transition-all font-mono border border-gray-700">
                    {val as React.ReactNode}
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
};

const StatCard = ({
  label,
  value,
  subtext,
  color = "gray",
}: {
  label: string;
  value: string | number;
  subtext?: string;
  color?: "emerald" | "amber" | "purple" | "cyan" | "gray";
}) => {
  const colorMap = {
    emerald: "text-emerald-400 border-emerald-500/20 bg-emerald-500/5",
    amber: "text-amber-400 border-amber-500/20 bg-amber-500/5",
    purple: "text-purple-400 border-purple-500/20 bg-purple-500/5",
    cyan: "text-cyan-400 border-cyan-500/20 bg-cyan-500/5",
    gray: "text-gray-400 border-gray-700/50 bg-gray-800/50",
  };

  return (
    <div
      className={`p-6 rounded-2xl border ${colorMap[color]} backdrop-blur-sm shadow-lg`}
    >
      <div className="text-gray-400 text-xs font-bold uppercase tracking-wider mb-2">
        {label}
      </div>
      <div className="text-3xl font-extrabold text-white tracking-tight">
        {value}
      </div>
      {subtext && (
        <div className="text-xs text-gray-500 mt-2 font-medium flex items-center gap-1">
          {subtext}
        </div>
      )}
    </div>
  );
};
