import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  RefreshCw,
  AlertTriangle,
  Info,
  CheckCircle2,
  BarChart2
} from 'lucide-react';
import { api } from '../api';
import StatusBadge from '../components/StatusBadge';

export default function ExplainabilityPage({ activeModel }) {
  const [loading, setLoading] = useState(false);
  const [explainData, setExplainData] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [sampleSize, setSampleSize] = useState(50);

  useEffect(() => {
    if (activeModel?.id) {
      loadExplainability(activeModel.id);
    }
  }, [activeModel?.id]);

  const loadExplainability = async (modelId) => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const data = await api.getExplainability(modelId, sampleSize);
      setExplainData(data);
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (!activeModel) {
    return (
      <div className="workspace-canvas">
        <div className="card" style={{ padding: 48, textAlign: 'center', color: 'var(--text-secondary)' }}>
          <AlertTriangle size={40} color="var(--warning)" style={{ marginBottom: 12 }} />
          <h3 style={{ fontWeight: 700, marginBottom: 6 }}>No Active Model for Explainability</h3>
          <p style={{ fontSize: '0.85rem', maxWidth: 440, margin: '0 auto 16px' }}>
            Please benchmark and activate a model to generate SHAP global feature attributions.
          </p>
        </div>
      </div>
    );
  }

  const featureImportance = explainData?.feature_importance || [];

  return (
    <div className="workspace-canvas">
      {/* Title */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: '1.4rem', fontWeight: 700 }}>Global Model Explainability (SHAP)</h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Shapley Additive Explanations — quantify feature importance and directional contributions to model predictions.
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button
            className="btn btn-secondary btn-sm"
            onClick={() => loadExplainability(activeModel.id)}
            disabled={loading}
          >
            <RefreshCw size={13} className={loading ? 'spin' : ''} />
            <span>Recalculate SHAP</span>
          </button>
        </div>
      </div>

      {errorMsg && (
        <div className="alert-banner danger">
          <AlertTriangle size={16} />
          <span>{errorMsg}</span>
        </div>
      )}

      {loading ? (
        <div className="card" style={{ padding: 60, textAlign: 'center', color: 'var(--text-secondary)' }}>
          <Sparkles size={36} color="var(--primary)" className="spin" style={{ marginBottom: 12 }} />
          <div style={{ fontWeight: 700, fontSize: '1.1rem' }}>Computing SHAP Shapley Values...</div>
          <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: 4 }}>
            Evaluating feature attributions across background distribution sample
          </div>
        </div>
      ) : explainData ? (
        <>
          {/* Explainer Metadata Card */}
          <div className="card" style={{ marginBottom: 24, borderLeft: '4px solid var(--primary)' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 14 }}>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Active Model</div>
                <div style={{ fontWeight: 700 }}>{explainData.algorithm}</div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Explainer Type</div>
                <div style={{ fontWeight: 700, color: 'var(--primary)' }}>{explainData.explainer_type}</div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Sample Evaluation Size</div>
                <div style={{ fontWeight: 700 }}>{explainData.sample_size} records</div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Task Type</div>
                <div style={{ fontWeight: 700, textTransform: 'capitalize' }}>{explainData.task_type}</div>
              </div>
            </div>

            {explainData.limitations && (
              <div className="alert-banner info" style={{ marginTop: 14, marginBottom: 0 }}>
                <Info size={16} />
                <span>{explainData.limitations}</span>
              </div>
            )}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 24 }}>
            {/* Feature Importance Horizontal Bars */}
            <div className="card">
              <div className="card-header">
                <div>
                  <div className="card-title">SHAP Global Importance Ranking</div>
                  <div className="card-subtitle">Mean absolute impact of each feature on model output</div>
                </div>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                {featureImportance.map((item) => (
                  <div key={item.feature}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: 4 }}>
                      <span style={{ fontWeight: 600 }}>{item.rank}. {item.feature}</span>
                      <span style={{ color: 'var(--text-secondary)' }}>
                        {item.importance}% (mean |SHAP| = {item.mean_abs_shap})
                      </span>
                    </div>
                    <div className="progress-container" style={{ margin: 0, height: 8 }}>
                      <div
                        className="progress-bar-fill"
                        style={{
                          width: `${Math.min(100, Math.max(5, item.importance))}%`,
                          background: item.rank <= 3 ? 'linear-gradient(90deg, #4f46e5, #06b6d4)' : '#94a3b8'
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Generated Summary Plot or Table */}
            <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
              <div className="card-header">
                <div className="card-title">SHAP Visual Summary Plot</div>
              </div>

              {explainData.summary_plot_base64 ? (
                <div style={{ textAlign: 'center', flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <img
                    src={`data:image/png;base64,${explainData.summary_plot_base64}`}
                    alt="SHAP Summary Plot"
                    style={{ maxWidth: '100%', maxHeight: 380, borderRadius: 'var(--radius-sm)' }}
                  />
                </div>
              ) : (
                <div className="table-responsive">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Rank</th>
                        <th>Feature</th>
                        <th>Impact %</th>
                        <th>Mean |SHAP|</th>
                      </tr>
                    </thead>
                    <tbody>
                      {featureImportance.map((it) => (
                        <tr key={it.feature}>
                          <td>#{it.rank}</td>
                          <td><strong>{it.feature}</strong></td>
                          <td>{it.importance}%</td>
                          <td>{it.mean_abs_shap}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
