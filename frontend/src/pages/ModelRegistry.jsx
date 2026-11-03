import React, { useState, useEffect } from 'react';
import {
  Package, CheckCircle, Cpu, Sparkles, ChevronRight,
  ShieldAlert, Target, TrendingUp, Clock, Trophy
} from 'lucide-react';
import { api } from '../api';

const FRAUD_METRICS = ['pr_auc', 'f2', 'recall', 'detection_rate', 'precision', 'f1', 'roc_auc', 'specificity'];
const METRIC_COLORS = {
  pr_auc: 'var(--purple-400)',
  f2: 'var(--orange-400)',
  recall: 'var(--cyan-400)',
  detection_rate: 'var(--cyan-400)',
  precision: 'var(--blue-400)',
  f1: 'var(--green-400)',
  roc_auc: 'var(--red-400)',
  specificity: 'var(--yellow-400)',
};

export default function ModelRegistry({ models, activeModel, onModelActivated, onNavigateToPredict }) {
  const [selectedModel, setSelectedModel] = useState(null);
  const [modelDetails, setModelDetails] = useState(null);
  const [activating, setActivating] = useState(false);

  useEffect(() => {
    const toLoad = activeModel || (models?.length > 0 ? models[0] : null);
    if (toLoad) {
      setSelectedModel(toLoad);
      loadDetails(toLoad.id);
    }
  }, [activeModel?.id, models?.length]);

  const loadDetails = async (id) => {
    try {
      const d = await api.getModel(id);
      setModelDetails(d);
    } catch (err) {
      console.error('Failed to load model details:', err);
    }
  };

  const handleActivate = async (id) => {
    setActivating(true);
    try {
      const updated = await api.activateModel(id);
      onModelActivated(updated);
      await loadDetails(id);
    } catch (err) {
      alert(`Activation error: ${err.message}`);
    } finally {
      setActivating(false);
    }
  };

  if (!models?.length) {
    return (
      <div className="page-content fade-in">
        <div
          className="card"
          style={{
            padding: 60, textAlign: 'center',
            display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12,
          }}
        >
          <Package size={40} style={{ opacity: 0.2 }} />
          <h3 style={{ fontWeight: 700 }}>No Trained Models Yet</h3>
          <p style={{ color: 'var(--text-muted)', maxWidth: 400, fontSize: 13 }}>
            Run a benchmark in <strong>Model Training</strong> to automatically train, evaluate, and persist fraud detection models.
          </p>
        </div>
      </div>
    );
  }

  const fraudMetrics = Object.entries(modelDetails?.metrics || {}).filter(
    ([k, v]) => FRAUD_METRICS.includes(k) && typeof v === 'number'
  );
  const otherMetrics = Object.entries(modelDetails?.metrics || {}).filter(
    ([k, v]) => !FRAUD_METRICS.includes(k) && typeof v === 'number' && !['cv_mean', 'cv_std'].includes(k)
  );

  return (
    <div className="page-content fade-in">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          {models.length} model{models.length !== 1 ? 's' : ''} registered
        </div>
        {activeModel && (
          <button className="btn btn-primary btn-sm" onClick={onNavigateToPredict}>
            Score Transactions <ChevronRight size={12} />
          </button>
        )}
      </div>

      <div className="grid-1-2">
        {/* Model list */}
        <div className="card" style={{ alignSelf: 'start' }}>
          <div className="card-title" style={{ marginBottom: 14 }}>
            <Package size={14} style={{ color: 'var(--blue-400)' }} />
            Registered Models
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {models.map((m) => {
              const isSelected = selectedModel?.id === m.id;
              const isActive = m.is_active;
              const prAuc = m.metrics?.pr_auc;
              return (
                <div
                  key={m.id}
                  onClick={() => { setSelectedModel(m); loadDetails(m.id); }}
                  style={{
                    padding: '12px 14px',
                    borderRadius: 8,
                    border: `1px solid ${isSelected ? 'rgba(239,68,68,0.3)' : 'var(--border)'}`,
                    background: isSelected ? 'rgba(239,68,68,0.06)' : 'var(--bg-elevated)',
                    cursor: 'pointer',
                    transition: 'all 0.15s',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ fontWeight: 700, fontSize: 13, color: isSelected ? 'var(--red-400)' : 'var(--text-primary)' }}>
                      {m.name}
                    </span>
                    <div style={{ display: 'flex', gap: 5 }}>
                      {m.is_tuned && <span className="badge badge-purple">Tuned</span>}
                      {isActive
                        ? <span className="badge badge-green">● Active</span>
                        : <span className="badge badge-gray">v{m.version}</span>
                      }
                    </div>
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', gap: 12 }}>
                    <span>{m.algorithm?.split(' ').slice(0, 3).join(' ')}</span>
                    {prAuc != null && (
                      <span style={{ color: 'var(--purple-400)', fontWeight: 600 }}>
                        PR-AUC: {prAuc.toFixed(4)}
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Model details */}
        {modelDetails ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Header */}
            <div className="card">
              <div className="card-header">
                <div>
                  <div className="card-title">
                    {modelDetails.is_active
                      ? <Trophy size={14} style={{ color: 'var(--yellow-400)' }} />
                      : <Cpu size={14} style={{ color: 'var(--blue-400)' }} />}
                    {modelDetails.name}
                    {modelDetails.is_active && <span className="badge badge-green" style={{ marginLeft: 8 }}>Active Model</span>}
                    {modelDetails.is_tuned && <span className="badge badge-purple" style={{ marginLeft: 6 }}>Tuned</span>}
                  </div>
                  <div className="card-subtitle">
                    ID: <span className="mono" style={{ fontSize: 11 }}>{modelDetails.id}</span>
                  </div>
                </div>
                {!modelDetails.is_active && (
                  <button
                    className="btn btn-primary"
                    onClick={() => handleActivate(modelDetails.id)}
                    disabled={activating}
                  >
                    <CheckCircle size={13} />
                    {activating ? 'Activating…' : 'Set Active'}
                  </button>
                )}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
                {[
                  { label: 'Algorithm', value: modelDetails.algorithm },
                  { label: 'Task Type', value: modelDetails.task_type },
                  { label: 'Target Column', value: modelDetails.target_column },
                  { label: 'Version', value: `v${modelDetails.version}` },
                  { label: 'Training Time', value: `${modelDetails.training_duration_seconds?.toFixed(1)}s` },
                  { label: 'MLflow Run', value: modelDetails.mlflow_run_id ? modelDetails.mlflow_run_id.slice(0, 12) + '…' : 'Local' },
                ].map(({ label, value }) => (
                  <div key={label} style={{ background: 'var(--bg-elevated)', padding: '10px 12px', borderRadius: 8 }}>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</div>
                    <div style={{ fontWeight: 600, fontSize: 12, color: 'var(--text-primary)' }}>{value}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Fraud detection metrics */}
            {fraudMetrics.length > 0 && (
              <div className="card">
                <div className="card-title" style={{ marginBottom: 12 }}>
                  <ShieldAlert size={14} style={{ color: 'var(--red-400)' }} />
                  Fraud Detection Metrics
                </div>
                <div className="metric-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
                  {fraudMetrics.map(([k, v]) => (
                    <div key={k} className="metric-card" style={{ '--metric-color': METRIC_COLORS[k] || 'var(--text-primary)', padding: '12px 14px' }}>
                      <div className="metric-card-label" style={{ fontSize: 10 }}>{k.replace(/_/g, ' ').toUpperCase()}</div>
                      <div className="metric-card-value" style={{ fontSize: 20 }}>
                        {k === 'recall' || k === 'precision' || k === 'specificity' || k === 'detection_rate'
                          ? `${(v * 100).toFixed(1)}%`
                          : v.toFixed(4)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Feature schema */}
            <div className="card">
              <div className="card-title" style={{ marginBottom: 10 }}>
                <Target size={14} style={{ color: 'var(--orange-400)' }} />
                Feature Schema ({modelDetails.feature_names?.length} features)
              </div>
              <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
                {modelDetails.feature_names?.map((f) => (
                  <span key={f} className="badge badge-gray" style={{ fontFamily: 'JetBrains Mono', fontSize: 10 }}>{f}</span>
                ))}
              </div>
            </div>

            {/* Hyperparameters */}
            <div className="card">
              <div className="card-title" style={{ marginBottom: 10 }}>
                <Sparkles size={14} style={{ color: 'var(--yellow-400)' }} />
                Hyperparameters
              </div>
              <pre
                style={{
                  background: 'var(--bg-base)', color: 'var(--text-code)',
                  padding: '10px 14px', borderRadius: 8,
                  fontSize: 11.5, fontFamily: 'JetBrains Mono',
                  maxHeight: 180, overflowY: 'auto', margin: 0,
                  border: '1px solid var(--border)',
                }}
              >
                {JSON.stringify(modelDetails.hyperparameters, null, 2)}
              </pre>
            </div>
          </div>
        ) : (
          <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 200, color: 'var(--text-muted)' }}>
            Select a model to view details
          </div>
        )}
      </div>
    </div>
  );
}
