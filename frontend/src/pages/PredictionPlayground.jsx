import React, { useState } from 'react';
import {
  Zap, ShieldAlert, ShieldCheck, AlertTriangle, Clock,
  BarChart2, RefreshCw, Info
} from 'lucide-react';
import { api } from '../api';

const RISK_META = {
  LOW:      { color: 'var(--risk-low)',      bg: 'rgba(74,222,128,0.07)',  icon: ShieldCheck,  label: 'LOW RISK' },
  MEDIUM:   { color: 'var(--risk-medium)',   bg: 'rgba(250,204,21,0.07)', icon: AlertTriangle, label: 'MEDIUM RISK' },
  HIGH:     { color: 'var(--risk-high)',     bg: 'rgba(251,146,60,0.08)', icon: AlertTriangle, label: 'HIGH RISK' },
  CRITICAL: { color: 'var(--risk-critical)', bg: 'rgba(239,68,68,0.10)',  icon: ShieldAlert,   label: 'CRITICAL RISK' },
};

function FraudGauge({ score }) {
  const pct = score * 100;
  const r = 54;
  const circ = 2 * Math.PI * r;
  const stroke = circ * score;
  const color = score < 0.3 ? 'var(--risk-low)' :
                score < 0.5 ? 'var(--risk-medium)' :
                score < 0.75 ? 'var(--risk-high)' : 'var(--risk-critical)';

  return (
    <div className="fraud-score-ring" style={{ width: 130, height: 130 }}>
      <svg width="130" height="130" viewBox="0 0 130 130" style={{ transform: 'rotate(-90deg)' }}>
        <circle cx="65" cy="65" r={r} fill="none" stroke="var(--bg-elevated)" strokeWidth="12" />
        <circle
          cx="65" cy="65" r={r}
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeDasharray={`${stroke} ${circ}`}
          strokeLinecap="round"
          style={{ transition: 'stroke-dasharray 0.8s ease, stroke 0.3s ease', filter: `drop-shadow(0 0 8px ${color})` }}
        />
      </svg>
      <div className="fraud-score-value" style={{ color }}>
        <span style={{ fontSize: 26, fontWeight: 800 }}>{pct.toFixed(0)}</span>
        <span style={{ fontSize: 11, fontWeight: 400, color: 'var(--text-muted)' }}>/ 100</span>
      </div>
    </div>
  );
}

const SAMPLE_TRANSACTIONS = [
  {
    label: '🟢 Legitimate Transaction',
    data: {
      amount: 45.99, hour_of_day: 14, day_of_week: 2,
      merchant_category: 'grocery', country: 'US', is_foreign_transaction: 0,
      velocity_last_1h: 1, velocity_last_24h: 3, account_age_days: 1200,
      distance_from_home_km: 5.2, days_since_last_transaction: 3,
      chip_used: 1, is_online_transaction: 0, card_type: 'credit',
      credit_utilization: 0.15, previous_fraud_flag: 0
    }
  },
  {
    label: '🔴 Suspicious Transaction',
    data: {
      amount: 1999.00, hour_of_day: 3, day_of_week: 6,
      merchant_category: 'online_retail', country: 'NG', is_foreign_transaction: 1,
      velocity_last_1h: 12, velocity_last_24h: 35, account_age_days: 14,
      distance_from_home_km: 4800.0, days_since_last_transaction: 0,
      chip_used: 0, is_online_transaction: 1, card_type: 'prepaid',
      credit_utilization: 0.97, previous_fraud_flag: 1
    }
  },
];

export default function PredictionPlayground({ activeModel, currentDataset, onPredictionLogged }) {
  const [jsonInput, setJsonInput] = useState(JSON.stringify(SAMPLE_TRANSACTIONS[0].data, null, 2));
  const [threshold, setThreshold] = useState(0.5);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const score = async () => {
    setLoading(true);
    setError(null);
    try {
      const parsed = JSON.parse(jsonInput);
      const res = await api.scoreTransaction(parsed, activeModel?.id || null, threshold);
      setResult(res);
      onPredictionLogged && onPredictionLogged();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const riskMeta = result ? RISK_META[result.risk_level] || RISK_META.MEDIUM : null;

  return (
    <div className="page-content fade-in">
      {!activeModel && (
        <div className="alert-banner warning" style={{ marginBottom: 16 }}>
          <AlertTriangle size={14} /> Activate a model in the registry first to score transactions.
        </div>
      )}

      <div className="grid-1-2">
        {/* Input panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="card">
            <div className="card-title" style={{ marginBottom: 12 }}>
              <Zap size={14} style={{ color: 'var(--yellow-400)' }} />
              Transaction Input
            </div>

            {/* Sample presets */}
            <div style={{ display: 'flex', gap: 6, marginBottom: 12, flexWrap: 'wrap' }}>
              {SAMPLE_TRANSACTIONS.map((s) => (
                <button
                  key={s.label}
                  className="btn btn-secondary btn-sm"
                  onClick={() => setJsonInput(JSON.stringify(s.data, null, 2))}
                >
                  {s.label}
                </button>
              ))}
            </div>

            <textarea
              className="form-textarea"
              value={jsonInput}
              onChange={(e) => setJsonInput(e.target.value)}
              style={{ minHeight: 340, fontFamily: 'JetBrains Mono', fontSize: 11.5, lineHeight: 1.7 }}
              spellCheck={false}
            />

            {/* Threshold slider */}
            <div style={{ marginTop: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                <label className="form-label" style={{ marginBottom: 0 }}>Decision Threshold</label>
                <span className="mono" style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{threshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.1" max="0.9" step="0.01"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--red-500)' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)' }}>
                <span>0.1 (Catch more fraud, more false alarms)</span>
                <span>0.9 (Very precise, miss more fraud)</span>
              </div>
            </div>

            {error && (
              <div className="alert-banner danger" style={{ marginTop: 12 }}>
                <AlertTriangle size={13} /> {error}
              </div>
            )}

            <button
              className="btn btn-primary"
              style={{ width: '100%', marginTop: 14, justifyContent: 'center' }}
              onClick={score}
              disabled={loading || !activeModel}
            >
              {loading ? <RefreshCw size={14} className="spin" /> : <Zap size={14} />}
              {loading ? 'Scoring…' : 'Score Transaction'}
            </button>
          </div>

          {/* Feature reference */}
          <div className="card">
            <div className="card-title" style={{ marginBottom: 10 }}>
              <Info size={14} style={{ color: 'var(--blue-400)' }} />
              Expected Features
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {(activeModel?.feature_names || Object.keys(SAMPLE_TRANSACTIONS[0].data)).map((f) => (
                <span key={f} className="badge badge-gray">{f}</span>
              ))}
            </div>
          </div>
        </div>

        {/* Result panel */}
        <div>
          {result ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {/* Main verdict */}
              <div
                className="card"
                style={{
                  background: riskMeta.bg,
                  borderColor: `${riskMeta.color}33`,
                  textAlign: 'center',
                }}
              >
                <div style={{ marginBottom: 20 }}>
                  <FraudGauge score={result.fraud_score} />
                </div>

                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, marginBottom: 8 }}>
                  <riskMeta.icon size={20} style={{ color: riskMeta.color }} />
                  <span
                    style={{
                      fontSize: 22,
                      fontWeight: 900,
                      color: riskMeta.color,
                      letterSpacing: '-0.02em',
                    }}
                  >
                    {riskMeta.label}
                  </span>
                </div>

                <div style={{ fontSize: 28, fontWeight: 900, marginBottom: 4, fontFamily: 'JetBrains Mono' }}>
                  <span style={{ color: result.predicted_fraud ? 'var(--risk-critical)' : 'var(--risk-low)' }}>
                    {result.predicted_fraud ? '⚠ FRAUD DETECTED' : '✓ LEGITIMATE'}
                  </span>
                </div>

                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  Fraud Probability: <strong className="mono" style={{ color: riskMeta.color }}>
                    {(result.fraud_score * 100).toFixed(2)}%
                  </strong>
                  {' '}· Threshold: <strong className="mono">{result.threshold_used}</strong>
                </div>
              </div>

              {/* Metric breakdown */}
              <div className="card">
                <div className="card-title" style={{ marginBottom: 12 }}>Score Details</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {[
                    { label: 'Fraud Score', value: `${(result.fraud_score * 100).toFixed(2)}%`, color: riskMeta.color },
                    { label: 'Risk Level', value: result.risk_level, color: riskMeta.color },
                    { label: 'Decision', value: result.predicted_fraud ? 'BLOCK / REVIEW' : 'APPROVE', color: result.predicted_fraud ? 'var(--risk-critical)' : 'var(--risk-low)' },
                    { label: 'Threshold Used', value: result.threshold_used?.toFixed(2), color: 'var(--text-primary)' },
                    { label: 'Latency', value: `${result.latency_ms?.toFixed(2)} ms`, color: 'var(--text-muted)' },
                    { label: 'Model', value: activeModel?.algorithm?.split(' ').slice(0, 3).join(' '), color: 'var(--blue-400)' },
                  ].map(({ label, value, color }) => (
                    <div key={label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
                      <span className="mono" style={{ fontWeight: 700, color }}>{value}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* SHAP explanations */}
              {result.explanation_top_features?.length > 0 && (
                <div className="card">
                  <div className="card-title" style={{ marginBottom: 12 }}>
                    <BarChart2 size={14} style={{ color: 'var(--purple-400)' }} />
                    Top Feature Contributions (SHAP)
                  </div>
                  {result.explanation_top_features.map((f) => {
                    const pct = Math.abs(f.shap_value) * 100;
                    const isPositive = f.shap_value > 0; // pushes towards fraud
                    return (
                      <div key={f.feature} style={{ marginBottom: 10 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 3 }}>
                          <span style={{ fontWeight: 600 }}>{f.feature}</span>
                          <span className="mono" style={{ color: isPositive ? 'var(--red-400)' : 'var(--green-400)' }}>
                            {isPositive ? '+' : ''}{f.shap_value.toFixed(4)}
                          </span>
                        </div>
                        <div className="progress-bar-container">
                          <div
                            className="progress-bar-fill"
                            style={{
                              width: `${Math.min(pct * 20, 100)}%`,
                              background: isPositive
                                ? 'linear-gradient(90deg, var(--red-600), var(--red-400))'
                                : 'linear-gradient(90deg, #15803d, var(--green-400))',
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              <div style={{ display: 'flex', gap: 8 }}>
                <button className="btn btn-secondary btn-sm" onClick={() => setResult(null)}>
                  <RefreshCw size={12} /> Clear
                </button>
              </div>
            </div>
          ) : (
            <div
              className="card"
              style={{
                height: '100%',
                minHeight: 400,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                textAlign: 'center',
                color: 'var(--text-muted)',
              }}
            >
              <ShieldAlert size={48} style={{ opacity: 0.15, marginBottom: 16 }} />
              <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 6 }}>Transaction Scoring Engine</div>
              <div style={{ fontSize: 12, maxWidth: 260 }}>
                Enter a transaction's feature values on the left and click "Score Transaction" to get a real-time fraud probability and risk classification.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
