import React from 'react';
import {
  ShieldAlert, TrendingUp, AlertTriangle, CheckCircle,
  Database, Cpu, Activity, ArrowRight, Zap, Target, BarChart2
} from 'lucide-react';
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis,
  ResponsiveContainer, Tooltip, Cell
} from 'recharts';

function MetricCard({ label, value, sub, color, icon: Icon, gradient }) {
  return (
    <div
      className="metric-card"
      style={{
        '--metric-color': color,
        '--metric-gradient': gradient,
        borderColor: color ? `${color}22` : undefined,
      }}
    >
      {Icon && (
        <div className="metric-card-icon">
          <Icon size={36} color={color || 'currentColor'} />
        </div>
      )}
      <div className="metric-card-label">{label}</div>
      <div className="metric-card-value">{value ?? '—'}</div>
      {sub && <div className="metric-card-sub">{sub}</div>}
    </div>
  );
}

const RISK_COLORS = {
  LOW: 'var(--risk-low)',
  MEDIUM: 'var(--risk-medium)',
  HIGH: 'var(--risk-high)',
  CRITICAL: 'var(--risk-critical)',
};

export default function OverviewDashboard({
  datasets = [],
  models = [],
  activeModel,
  monitoringSummary,
  onNavigate,
}) {
  const metrics = activeModel?.metrics || {};
  const prAuc = metrics.pr_auc ?? metrics.roc_auc;
  const f2 = metrics.f2;
  const recall = metrics.recall ?? metrics.detection_rate;
  const precision = metrics.precision;

  // Real-time fraud rate trend and risk distribution from monitoring telemetry
  const trendData = monitoringSummary?.fraud_rate_trend?.length > 0
    ? monitoringSummary.fraud_rate_trend
    : [
        { hour: '00:00', transactions: monitoringSummary?.total_predictions || 0, fraud: 0 }
      ];

  const riskDist = monitoringSummary?.risk_distribution || {};
  const riskData = [
    { level: 'LOW',      count: riskDist.LOW || 0, fill: 'var(--risk-low)' },
    { level: 'MEDIUM',   count: riskDist.MEDIUM || 0, fill: 'var(--risk-medium)' },
    { level: 'HIGH',     count: riskDist.HIGH || 0, fill: 'var(--risk-high)' },
    { level: 'CRITICAL', count: riskDist.CRITICAL || 0, fill: 'var(--risk-critical)' },
  ];

  const steps = [
    { id: 'datasets', label: '1. Ingest Data',       done: datasets.length > 0 },
    { id: 'studio',   label: '2. Train Models',      done: models.length > 0 },
    { id: 'registry', label: '3. Activate Model',    done: !!activeModel },
    { id: 'predict',  label: '4. Score Transactions',done: !!activeModel },
    { id: 'monitor',  label: '5. Monitor & Retrain', done: !!monitoringSummary },
  ];

  return (
    <div className="page-content fade-in">
      {/* Hero banner */}
      <div
        className="card section-gap"
        style={{
          background: 'linear-gradient(135deg, #0f1e35 0%, #1a0a0a 100%)',
          border: '1px solid rgba(239,68,68,0.15)',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            position: 'absolute', inset: 0,
            backgroundImage: 'radial-gradient(circle at 80% 50%, rgba(239,68,68,0.08) 0%, transparent 60%)',
            pointerEvents: 'none',
          }}
        />
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
              <ShieldAlert size={22} color="var(--red-400)" />
              <h2 style={{ fontSize: 20, fontWeight: 800, letterSpacing: '-0.02em' }}>
                Fraud Detection Command Center
              </h2>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: 13, maxWidth: 520 }}>
              End-to-end ML pipeline for financial transaction fraud detection.
              Upload transaction data, train imbalance-aware models, optimize thresholds, and monitor in production.
            </p>
          </div>
          {activeModel && (
            <div style={{ textAlign: 'right', flexShrink: 0 }}>
              <div className="badge badge-green" style={{ marginBottom: 6 }}>● Active Model</div>
              <div style={{ fontSize: 13, fontWeight: 700 }}>{activeModel.algorithm}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>v{activeModel.version} · {activeModel.target_column}</div>
            </div>
          )}
        </div>
      </div>

      {/* Workflow steps */}
      <div className="card section-gap">
        <div className="card-title" style={{ marginBottom: 14 }}>
          <Target size={15} style={{ color: 'var(--red-400)' }} />
          Detection Pipeline
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {steps.map((step) => (
            <button
              key={step.id}
              className="btn btn-secondary btn-sm"
              style={{
                borderColor: step.done ? 'rgba(74,222,128,0.3)' : undefined,
                color: step.done ? 'var(--green-400)' : undefined,
              }}
              onClick={() => onNavigate(step.id)}
            >
              {step.done ? <CheckCircle size={12} /> : <ArrowRight size={12} />}
              {step.label}
            </button>
          ))}
        </div>
      </div>

      {/* Key metrics */}
      <div className="metric-grid section-gap">
        <MetricCard
          label="PR-AUC Score"
          value={prAuc != null ? prAuc.toFixed(4) : '—'}
          sub="Primary fraud metric"
          color="var(--red-400)"
          icon={ShieldAlert}
        />
        <MetricCard
          label="F2 Score"
          value={f2 != null ? f2.toFixed(4) : '—'}
          sub="Recall-weighted"
          color="var(--orange-400)"
          icon={Target}
        />
        <MetricCard
          label="Detection Rate"
          value={recall != null ? `${(recall * 100).toFixed(1)}%` : '—'}
          sub="Fraud recall"
          color="var(--cyan-400)"
          icon={TrendingUp}
        />
        <MetricCard
          label="Precision"
          value={precision != null ? `${(precision * 100).toFixed(1)}%` : '—'}
          sub="At 0.5 threshold"
          color="var(--purple-400)"
          icon={CheckCircle}
        />
        <MetricCard
          label="Datasets"
          value={datasets.length}
          sub="Loaded"
          color="var(--blue-400)"
          icon={Database}
        />
        <MetricCard
          label="Models Trained"
          value={models.length}
          sub="In registry"
          color="var(--green-400)"
          icon={Cpu}
        />
      </div>

      {/* Charts row */}
      <div className="grid-2 section-gap">
        {/* Fraud Rate Trend */}
        <div className="card">
          <div className="card-title" style={{ marginBottom: 14 }}>
            <Activity size={14} style={{ color: 'var(--red-400)' }} />
            Simulated Transaction Volume
          </div>
          <ResponsiveContainer width="100%" height={160}>
            <AreaChart data={trendData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="fraudGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="txnGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#60a5fa" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#60a5fa" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="hour" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
              <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
              <Tooltip
                contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', fontSize: 12 }}
                labelStyle={{ color: 'var(--text-secondary)' }}
              />
              <Area type="monotone" dataKey="transactions" stroke="var(--blue-400)" fill="url(#txnGrad)" strokeWidth={1.5} name="All Transactions" />
              <Area type="monotone" dataKey="fraud" stroke="var(--red-400)" fill="url(#fraudGrad)" strokeWidth={2} name="Fraud Alerts" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Level Distribution */}
        <div className="card">
          <div className="card-title" style={{ marginBottom: 14 }}>
            <BarChart2 size={14} style={{ color: 'var(--orange-400)' }} />
            Risk Level Distribution
          </div>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={riskData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
              <XAxis dataKey="level" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
              <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
              <Tooltip
                contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', fontSize: 12 }}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]} name="Transactions">
                {riskData.map((entry, index) => (
                  <Cell key={index} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* System status */}
      <div className="grid-2">
        {/* Monitoring card */}
        <div className="card">
          <div className="card-title" style={{ marginBottom: 12 }}>
            <Activity size={14} style={{ color: 'var(--cyan-400)' }} />
            Monitoring Status
          </div>
          {monitoringSummary ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                <span style={{ color: 'var(--text-secondary)' }}>Total Predictions</span>
                <span className="mono" style={{ color: 'var(--text-primary)' }}>{monitoringSummary.total_predictions}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                <span style={{ color: 'var(--text-secondary)' }}>Drift Detected</span>
                <span className={monitoringSummary.drift_detected ? 'badge badge-red' : 'badge badge-green'}>
                  {monitoringSummary.drift_detected ? 'YES' : 'NONE'}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                <span style={{ color: 'var(--text-secondary)' }}>Drifted Features</span>
                <span className="mono" style={{ color: monitoringSummary.drifted_features_count > 0 ? 'var(--orange-400)' : 'var(--green-400)' }}>
                  {monitoringSummary.drifted_features_count} / {monitoringSummary.total_features_monitored}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                <span style={{ color: 'var(--text-secondary)' }}>Avg Latency</span>
                <span className="mono" style={{ color: 'var(--text-primary)' }}>{monitoringSummary.mean_latency_ms?.toFixed(1)} ms</span>
              </div>
            </div>
          ) : (
            <div style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: '16px 0' }}>
              Activate a model and score some transactions to see monitoring data.
            </div>
          )}
        </div>

        {/* Quick actions */}
        <div className="card">
          <div className="card-title" style={{ marginBottom: 12 }}>
            <Zap size={14} style={{ color: 'var(--yellow-400)' }} />
            Quick Actions
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <button className="btn btn-primary" onClick={() => onNavigate('datasets')} style={{ justifyContent: 'flex-start' }}>
              <Database size={14} /> Upload Transaction Data
            </button>
            <button className="btn btn-secondary" onClick={() => onNavigate('studio')} style={{ justifyContent: 'flex-start' }}>
              <Cpu size={14} /> Train Fraud Models
            </button>
            <button className="btn btn-secondary" onClick={() => onNavigate('predict')} style={{ justifyContent: 'flex-start' }}>
              <Zap size={14} /> Score Transactions
            </button>
            <button className="btn btn-secondary" onClick={() => onNavigate('threshold')} style={{ justifyContent: 'flex-start' }}>
              <Target size={14} /> Optimize Threshold
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
