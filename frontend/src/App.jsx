import React, { useState } from "react";
import { analyzeLogs } from "./api";
import "./index.css";

export default function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const onSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);

    if (!file) {
      setError("Please select a CSV log file.");
      return;
    }

    try {
      setLoading(true);
      const data = await analyzeLogs(file);
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to analyze logs.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="hero">
        <div className="hero-badge">Log Anomaly Studio</div>
        <h1>Detect anomalies, then explain them with LLM context.</h1>
        <p>
          Upload your structured log CSV and get LogBERT anomaly scores plus a
          LLM verdict alongside possible solutions
        </p>
      </header>

      <section className="panel">
        <form onSubmit={onSubmit} className="upload">
          <label className="file">
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            <span>{file ? file.name : "Choose CSV log file"}</span>
          </label>
          <button type="submit" disabled={loading}>
            {loading ? "Analyzing..." : "Run Analysis"}
          </button>
        </form>
        {error && <div className="error">{error}</div>}
      </section>

      {result && (
        <section className="results">
          <div className="summary">
            <div>
              <h3>Summary</h3>
              <div className="summary-grid">
                <div>
                  <span>Total sequences</span>
                  <strong>{result.total_sequences}</strong>
                </div>
                <div>
                  <span>Skipped</span>
                  <strong>{result.skipped_sequences}</strong>
                </div>
                <div>
                  <span>Anomalies</span>
                  <strong>{result.anomalies_found}</strong>
                </div>
              </div>
            </div>
            <div className="file-pill">{result.file}</div>
          </div>

          <div className="cards">
            {result.results.length === 0 && (
              <div className="empty">No anomalies above the threshold.</div>
            )}
            {result.results.map((item) => (
              <article className="card" key={item.sequence_id}>
                <div className="card-head">
                  <div>
                    <h4>Sequence {item.sequence_id}</h4>
                    <span>Score: {item.anomaly_score.toFixed(4)}</span>
                  </div>
                  <div className={`verdict ${item.llm_verdict}`}>
                    {item.llm_verdict}
                  </div>
                </div>
                <div className="card-body">
                  <div className="log-seq">
                    {item.event_templates.map((line, idx) => (
                      <div key={idx}>{line}</div>
                    ))}
                  </div>
                  <div className="llm">
                    <div className="llm-title">LLM response</div>
                    {item.llm_explanation && (
                      <div className="llm-block">
                        <div className="llm-label">Explanation</div>
                        <div className="llm-text">{item.llm_explanation}</div>
                      </div>
                    )}
                    {item.llm_solution && (
                      <div className="llm-block">
                        <div className="llm-label">Solution</div>
                        <div className="llm-text">{item.llm_solution}</div>
                      </div>
                    )}
                    {!item.llm_explanation && !item.llm_solution && (
                      <pre>{item.llm_response}</pre>
                    )}
                  </div>
                </div>
              </article>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
