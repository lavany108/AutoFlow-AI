import React, { useState } from 'react';
import {
  Sliders, Target, DollarSign, ShieldAlert, CheckCircle, AlertTriangle
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, ResponsiveContainer,
  Tooltip, ReferenceLine, CartesianGrid
} from 'recharts';
import { api } from '../api';

const STRATEGIES = [
  { value: 'f2', label: 'Maximize F2 Score', desc: 'Best balance of recall (catching fraud) over precision. Recommended for most fraud use cases.' },
  { value: 'f1', label: 'Maximize F1 Score', desc: 'Equal weight to precision and recall.' },
  { value: 'cost', label: 'Minimize Business Cost', desc: 'Minimize total cost of false alarms + missed fraud (requires cost parameters).' },
  { value: 'precision_at_recall', label: 'Max Precision @ Min Recall', desc: 'Highest precision subject to recall ≥ minimum recall floor.' },
];

function CostInput({ label, value, onChange, prefix = '$' }) {
  return (
    <div className="form-group">
      <label className="form-label">{label}</label>
      <div style={{ display: 'flex', alignItems: 'center', gap: 0 }}>
        <span style={{
          padding: '9px 10px', background: 'var(--bg-elevated)',
          border: '1px solid var(--border)', borderRight: 'none',
          borderRadius: '6px 0 0 6px', fontSize: 12, color: 'var(--text-muted)'
        }}>{prefix}</span>
        <input
          type="number"
          className="form-input"
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
          style={{ borderRadius: '0 6px 6px 0', borderLeft: 'none' }}
          min={0}
          step={10}
        />
      </div>
    </div>
  );
}

export default function ThresholdOptimizerPage({ activeModel }) {
  const [strategy, setStrategy] = useState('f2');
  const [minRecall, setMinRecall] = useState(0.80);
  const [costFP, setCostFP] = useState(10);
  const [costFN, setCostFN] = useState(500);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const optimize = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.optimizeThreshold({
        model_id: activeModel?.id || null,
        strategy,
        min_recall: minRecall,
        cost_fp: costFP,
        cost_fn: costFN,
      });
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const chartData = result?.threshold_scores
    ?.filter((t) => t.score !== null)
    ?.slice(0, 80)
    .map((t) => ({
      threshold: t.threshold,
      precision: t.precision,
      recall: t.recall,
      score: t.score != null && isFinite(t.score) ? Math.max(0, Math.min(t.score, 1)) : null,
    })) || [];

  const selectedStrategy = STRATEGIES.find((s) => s.value === strategy);

  return (
    <div className="page-content fade-in">
      {!activeModel && (
        <div className="alert-banner warning" style={{ marginBottom: 16 }}>
          <AlertTriangle size={14} /> Activate a model in the registry first to run threshold optimization.
        </div>
      )}

      <div className="grid-2">
        {/* Config panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="card">
            <div className="card-title" style={{ marginBottom: 14 }}>
              <Sliders size={14} style={{ color: 'var(--purple-400)' }} />
              Optimization Strategy
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 16 }}>
              {STRATEGIES.map((s) => (
                <label
                  key={s.value}
                  style={{
                    padding: '10px 12px',
                    background: strategy === s.value ? 'rgba(239,68,68,0.08)' : 'var(--bg-elevated)',
                    border: `1px solid ${strategy === s.value ? 'rgba(239,68,68,0.25)' : 'var(--border)'}`,
                    borderRadius: 8,
                    cursor: 'pointer',
                    transition: 'all 0.15s',
                  }}
                  onClick={() => setStrategy(s.value)}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <input
                      type="radio"
                      name="strategy"
                      value={s.value}
                      checked={strategy === s.value}
                      onChange={() => setStrategy(s.value)}
                      style={{ accentColor: 'var(--red-500)', flexShrink: 0 }}
                    />
                    <span style={{ fontWeight: 600, fontSize: 13, color: strategy === s.value ? 'var(--red-400)' : 'var(--text-primary)' }}>
                      {s.label}
                    </span>
                  </div>
                  <p style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 22, lineHeight: 1.5 }}>
                    {s.desc}
                  </p>
                </label>
              ))}
            </div>
          </div>

          <div className="card">
            <div className="card-title" style={{ marginBottom: 14 }}>
              <DollarSign size={14} style={{ color: 'var(--green-400)' }} />
              Business Cost Parameters
            </div>
            <CostInput label="Cost per False Positive (analyst time)" value={costFP} onChange={setCostFP} />
            <CostInput label="Cost per False Negative (missed fraud)" value={costFN} onChange={setCostFN} />
            {strategy === 'precision_at_recall' && (
              <div className="form-group">
                <label className="form-label">Minimum Recall Floor</label>
                <input
                  type="range"
                  min="0.5" max="0.99" step="0.01"
                  value={minRecall}
                  onChange={(e) => setMinRecall(parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--red-500)' }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)' }}>
                  <span>50%</span>
                  <span style={{ fontWeight: 700, color: 'var(--red-400)' }}>{(minRecall * 100).toFixed(0)}%</span>
                  <span>99%</span>
                </div>
              </div>
            )}

            {error && (
              <div className="alert-banner danger">
                <AlertTriangle size={13} /> {error}
              </div>
            )}

            <button
              className="btn btn-primary"
              style={{ width: '100%', marginTop: 8, justifyContent: 'center' }}
              onClick={optimize}
              disabled={loading || !activeModel}
            >
              {loading ? <Target size={14} className="spin" /> : <Target size={14} />}
              {loading ? 'Optimizing…' : 'Find Optimal Threshold'}
            </button>
          </div>
        </div>

        {/* Results panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {result ? (
            <>
              {/* Optimal threshold highlight */}
              <div
                className="card"
                style={{
                  background: 'linear-gradient(135deg, rgba(239,68,68,0.08), rgba(185,28,28,0.05))',
                  borderColor: 'rgba(239,68,68,0.2)',
                  textAlign: 'center',
                }}
              >
                <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 8 }}>
                  Optimal Threshold ({selectedStrategy?.label})
                </div>
                <div style={{ fontSize: 52, fontWeight: 900, letterSpacing: '-0.03em', color: 'var(--red-400)', fontFamily: 'JetBrains Mono', lineHeight: 1 }}>
                  {result.optimal_threshold.toFixed(3)}
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>
                  Use this value as the decision boundary in your production scoring service
                </div>
              </div>

              {/* Metric summary */}
              <div className="metric-grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)' }}>
                {[
                  { label: 'Precision', value: (result.optimal_precision * 100).toFixed(1) + '%', color: 'var(--purple-400)' },
                  { label: 'Recall', value: (result.optimal_recall * 100).toFixed(1) + '%', color: 'var(--cyan-400)' },
                  { label: 'F1 Score', value: result.optimal_f1.toFixed(4), color: 'var(--blue-400)' },
                  { label: 'F2 Score', value: result.optimal_f2.toFixed(4), color: 'var(--orange-400)' },
                ].map(({ label, value, color }) => (
                  <div key={label} className="metric-card" style={{ '--metric-color': color }}>
                    <div className="metric-card-label">{label}</div>
                    <div className="metric-card-value">{value}</div>
                  </div>
                ))}
              </div>

              {/* Confusion matrix at optimal threshold */}
              <div className="card">
                <div className="card-title" style={{ marginBottom: 12 }}>
                  Confusion Matrix at Optimal Threshold
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  {[
                    { label: 'True Positives', sub: 'Fraud correctly caught', value: result.tp, color: 'var(--green-400)' },
                    { label: 'False Positives', sub: 'False alarms', value: result.fp, color: 'var(--orange-400)' },
                    { label: 'False Negatives', sub: 'Missed fraud', value: result.fn, color: 'var(--red-400)' },
                    { label: 'True Negatives', sub: 'Legit correctly approved', value: result.tn, color: 'var(--blue-400)' },
                  ].map(({ label, sub, value, color }) => (
                    <div key={label} style={{ background: 'var(--bg-elevated)', borderRadius: 8, padding: '12px 14px', borderLeft: `3px solid ${color}` }}>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 2 }}>{label}</div>
                      <div style={{ fontSize: 22, fontWeight: 800, color, fontFamily: 'JetBrains Mono' }}>{value?.toLocaleString()}</div>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{sub}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Business impact */}
              {result.business_impact && (
                <div className="card">
                  <div className="card-title" style={{ marginBottom: 12 }}>
                    <DollarSign size={14} style={{ color: 'var(--green-400)' }} />
                    Business Impact
                  </div>
                  {[
                    { label: 'Estimated Annual Savings', value: `$${result.business_impact.estimated_savings?.toLocaleString()}`, color: 'var(--green-400)' },
                    { label: 'Baseline Cost (no model)', value: `$${result.business_impact.baseline_cost_no_model?.toLocaleString()}`, color: 'var(--text-muted)' },
                    { label: 'Operational Cost', value: `$${result.business_impact.total_operational_cost?.toLocaleString()}`, color: 'var(--orange-400)' },
                    { label: 'Fraud Amount Caught', value: `$${result.business_impact.fraud_amount_caught?.toLocaleString()}`, color: 'var(--cyan-400)' },
                  ].map(({ label, value, color }) => (
                    <div key={label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 10 }}>
                      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
                      <span className="mono" style={{ fontWeight: 700, color }}>{value}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Precision-Recall curve */}
              {chartData.length > 0 && (
                <div className="card">
                  <div className="card-title" style={{ marginBottom: 12 }}>
                    Threshold Sensitivity Curve
                  </div>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={chartData} margin={{ top: 0, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="threshold" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                      <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                      <Tooltip
                        contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', fontSize: 11 }}
                        labelFormatter={(v) => `Threshold: ${v}`}
                      />
                      <ReferenceLine x={result.optimal_threshold} stroke="var(--red-400)" strokeDasharray="4 2" label={{ value: 'Optimal', fill: 'var(--red-400)', fontSize: 10 }} />
                      <Line type="monotone" dataKey="precision" stroke="var(--purple-400)" strokeWidth={2} dot={false} name="Precision" />
                      <Line type="monotone" dataKey="recall" stroke="var(--cyan-400)" strokeWidth={2} dot={false} name="Recall" />
                      <Line type="monotone" dataKey="score" stroke="var(--orange-400)" strokeWidth={2} dot={false} name="Objective" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          ) : (
            <div
              className="card"
              style={{
                minHeight: 400, display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center', textAlign: 'center',
                color: 'var(--text-muted)',
              }}
            >
              <Sliders size={48} style={{ opacity: 0.15, marginBottom: 16 }} />
              <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 6 }}>Threshold Optimizer</div>
              <div style={{ fontSize: 12, maxWidth: 280 }}>
                Select an optimization strategy on the left, configure business cost parameters, and click "Find Optimal Threshold" to compute the best decision boundary for your fraud model.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
