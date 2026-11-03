import React, { useState, useEffect } from 'react';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  Clock,
  Flame,
  ArrowRight,
  TrendingDown,
  ShieldAlert,
  RotateCcw,
  Send
} from 'lucide-react';
import { api } from '../api';
import MetricCard from '../components/MetricCard';
import StatusBadge from '../components/StatusBadge';

export default function MonitoringCenter({
  activeModel,
  monitoringSummary,
  onRefreshMonitoring,
  onModelRetrained
}) {
  const [retrainJobs, setRetrainJobs] = useState([]);
  const [isRetraining, setIsRetraining] = useState(false);
  const [retrainTolerance, setRetrainTolerance] = useState(0.05);
  const [retrainMsg, setRetrainMsg] = useState(null);

  // Ground truth submission
  const [logIdInput, setLogIdInput] = useState('');
  const [actualLabelInput, setActualLabelInput] = useState('');
  const [outcomeSubmitting, setOutcomeSubmitting] = useState(false);
  const [outcomeSuccess, setOutcomeSuccess] = useState(false);

  useEffect(() => {
    loadRetrainJobs();
  }, []);

  const loadRetrainJobs = async () => {
    try {
      const jobs = await api.getRetrainJobs();
      setRetrainJobs(jobs || []);
    } catch (err) {
      console.error('Failed to load retrain jobs:', err);
    }
  };

  const handleTriggerRetrain = async (reason = 'Manual UI Trigger') => {
    if (!activeModel?.id) return;
    setIsRetraining(true);
    setRetrainMsg(null);
    try {
      const res = await api.triggerRetraining(activeModel.id, reason, retrainTolerance);
      setRetrainMsg(res.message);
      // Poll retrain job
      const pollInterval = setInterval(async () => {
        try {
          const job = await api.getRetrainJob(res.job_id);
          if (['COMPLETED', 'FAILED'].includes(job.status)) {
            clearInterval(pollInterval);
            setIsRetraining(false);
            await loadRetrainJobs();
            onModelRetrained();
          }
        } catch (_) {}
      }, 2000);
    } catch (err) {
      alert(`Error triggering retraining: ${err.message}`);
      setIsRetraining(false);
    }
  };

  const handleSubmitOutcome = async (e) => {
    e.preventDefault();
    if (!logIdInput || actualLabelInput === '') return;
    setOutcomeSubmitting(true);
    setOutcomeSuccess(false);
    try {
      await api.submitActualOutcome(logIdInput, actualLabelInput);
      setOutcomeSuccess(true);
      setLogIdInput('');
      setActualLabelInput('');
      onRefreshMonitoring();
      setTimeout(() => setOutcomeSuccess(false), 3000);
    } catch (err) {
      alert(`Error submitting ground truth: ${err.message}`);
    } finally {
      setOutcomeSubmitting(false);
    }
  };

  const featureDrift = monitoringSummary?.feature_drift || [];
  const hasDrift = monitoringSummary?.drift_detected || false;

  return (
    <div className="workspace-canvas">
      {/* Title */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: '1.4rem', fontWeight: 700 }}>Production Monitoring & Retraining</h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Real-time inference latency tracking, Kolmogorov-Smirnov & Chi-Square feature drift tests, and autonomous retrain workflows.
          </p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={onRefreshMonitoring}>
          <RefreshCw size={13} />
          <span>Refresh Telemetry</span>
        </button>
      </div>

      {/* Drift Warning Banner if Drift Detected */}
      {hasDrift && (
        <div className="alert-banner warning" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <AlertTriangle size={20} />
            <div>
              <strong>Input Feature Drift Detected:</strong> {monitoringSummary?.drifted_features_count} out of {monitoringSummary?.total_features_monitored} monitored features show significant distribution shifts.
            </div>
          </div>
          <button
            className="btn btn-primary btn-sm"
            onClick={() => handleTriggerRetrain('Automatic Retrain Triggered by Statistical Drift')}
            disabled={isRetraining}
          >
            <RotateCcw size={14} />
            <span>{isRetraining ? 'Retraining...' : 'Retrain Active Model'}</span>
          </button>
        </div>
      )}

      {/* Telemetry Metric Cards */}
      <div className="metrics-grid">
        <MetricCard
          title="Total Inferences Logged"
          value={monitoringSummary?.total_predictions || 0}
          subtitle="Requests captured in telemetry"
          icon={Activity}
          color="indigo"
          badge={<StatusBadge status="INFO" label="Telemetry Active" />}
        />
        <MetricCard
          title="Average Latency"
          value={`${(monitoringSummary?.mean_latency_ms || 0).toFixed(1)} ms`}
          subtitle="Real-time execution latency"
          icon={Clock}
          color="cyan"
          badge={<StatusBadge status="INFO" label="Sub-second" />}
        />
        <MetricCard
          title="P95 Latency"
          value={`${(monitoringSummary?.p95_latency_ms || 0).toFixed(1)} ms`}
          subtitle="95th percentile response"
          icon={Clock}
          color="purple"
          badge={<StatusBadge status="INFO" label="SLA Standard" />}
        />
        <MetricCard
          title="Feature Drift Status"
          value={hasDrift ? 'Drift Alert' : 'Stable'}
          subtitle={`${monitoringSummary?.drifted_features_count || 0} drifted features`}
          icon={ShieldAlert}
          color={hasDrift ? 'danger' : 'success'}
          badge={<StatusBadge status={hasDrift ? 'DRIFT DETECTED' : 'STABLE'} />}
        />
      </div>

      {/* Statistical Feature Drift Inspection Table */}
      <div className="card" style={{ marginBottom: 28 }}>
        <div className="card-header">
          <div>
            <div className="card-title">Statistical Feature Drift Analysis</div>
            <div className="card-subtitle">
              KS-Test (numerical features) & Chi-Square test (categorical features) against training baseline (threshold p &lt; 0.05)
            </div>
          </div>
        </div>

        {featureDrift.length === 0 ? (
          <div style={{ padding: 32, textAlign: 'center', color: 'var(--text-muted)' }}>
            No feature drift telemetry calculated yet. Run single or batch predictions in the Prediction Playground to test incoming distributions.
          </div>
        ) : (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Type</th>
                  <th>Statistical Test</th>
                  <th>Statistic Value</th>
                  <th>P-Value</th>
                  <th>Threshold</th>
                  <th>Status</th>
                  <th>Reference vs Current</th>
                </tr>
              </thead>
              <tbody>
                {featureDrift.map((item) => (
                  <tr key={item.feature} style={{ backgroundColor: item.is_drifted ? 'rgba(239, 68, 68, 0.05)' : undefined }}>
                    <td><strong>{item.feature}</strong></td>
                    <td><StatusBadge status="INFO" label={item.feature_type} /></td>
                    <td>{item.statistic_name}</td>
                    <td>{item.statistic_value.toFixed(4)}</td>
                    <td style={{ fontWeight: item.is_drifted ? 700 : 400, color: item.is_drifted ? 'var(--danger)' : 'inherit' }}>
                      {item.p_value.toFixed(4)}
                    </td>
                    <td>{item.threshold}</td>
                    <td>
                      <StatusBadge status={item.is_drifted ? 'DRIFT DETECTED' : 'STABLE'} />
                    </td>
                    <td style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      Ref: {String(item.ref_mean_or_mode)} | Curr: {String(item.curr_mean_or_mode)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Concept Drift & Ground Truth Outcome Interface */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 24, marginBottom: 28 }}>
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Concept Drift Verification</div>
              <div className="card-subtitle">
                Compare delayed true outcomes with predictions to detect real degradation
              </div>
            </div>
            <StatusBadge
              status={monitoringSummary?.concept_drift_status?.toUpperCase() || 'UNKNOWN'}
              label={monitoringSummary?.concept_drift_status === 'unknown' ? 'No Ground Truth' : monitoringSummary?.concept_drift_status}
            />
          </div>

          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: 14 }}>
            Input feature distributions alone do not establish concept drift without actual outcomes. Provide true labels for past predictions to audit model performance degradation.
          </div>

          <form onSubmit={handleSubmitOutcome} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr auto', gap: 12, alignItems: 'end' }}>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Prediction Log ID</label>
              <input
                type="text"
                className="form-control"
                placeholder="Paste prediction ID..."
                value={logIdInput}
                onChange={(e) => setLogIdInput(e.target.value)}
                required
              />
            </div>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Actual Outcome</label>
              <input
                type="text"
                className="form-control"
                placeholder="True label or value..."
                value={actualLabelInput}
                onChange={(e) => setActualLabelInput(e.target.value)}
                required
              />
            </div>
            <button
              type="submit"
              className="btn btn-secondary"
              disabled={outcomeSubmitting}
            >
              <Send size={14} />
              <span>{outcomeSubmitting ? 'Submitting...' : 'Record'}</span>
            </button>
          </form>

          {outcomeSuccess && (
            <div className="alert-banner success" style={{ marginTop: 14, marginBottom: 0 }}>
              <CheckCircle2 size={16} />
              <span>Ground truth outcome recorded for concept drift evaluation.</span>
            </div>
          )}
        </div>

        {/* Autonomous Retraining Trigger Card */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div className="card-header">
              <div className="card-title">Trigger Retraining Pipeline</div>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: 14 }}>
              Trains a fresh candidate model on updated data, validates metrics against the active model, and safely promotes it only if it satisfies your tolerance threshold.
            </p>

            <div className="form-group">
              <label className="form-label">
                Performance Tolerance Margin: {Math.round(retrainTolerance * 100)}%
              </label>
              <input
                type="range"
                min="0.01"
                max="0.15"
                step="0.01"
                value={retrainTolerance}
                onChange={(e) => setRetrainTolerance(parseFloat(e.target.value))}
              />
              <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                Candidate must score at least (Active Model - {Math.round(retrainTolerance * 100)}%) to be promoted.
              </span>
            </div>
          </div>

          <button
            className="btn btn-primary"
            onClick={() => handleTriggerRetrain('Manual User Retrain Trigger')}
            disabled={isRetraining || !activeModel}
            style={{ width: '100%', marginTop: 10 }}
          >
            <RotateCcw size={16} />
            <span>{isRetraining ? 'Retraining Pipeline Running...' : 'Execute Autonomous Retrain Job'}</span>
          </button>
        </div>
      </div>

      {/* Retrain Jobs Audit Log */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Autonomous Retraining Audit History</div>
            <div className="card-subtitle">Historical retraining triggers, candidate evaluations, and promotion decisions</div>
          </div>
        </div>

        {retrainJobs.length === 0 ? (
          <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-muted)' }}>
            No autonomous retraining jobs executed yet. Trigger a retraining job above to record audit logs.
          </div>
        ) : (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Job ID</th>
                  <th>Trigger Reason</th>
                  <th>Status</th>
                  <th>Promoted to Active?</th>
                  <th>Decision Rationale</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {retrainJobs.map((job) => (
                  <tr key={job.id}>
                    <td><code style={{ fontSize: '0.78rem' }}>{job.id.slice(0, 8)}...</code></td>
                    <td>{job.triggered_by}</td>
                    <td><StatusBadge status={job.status} /></td>
                    <td>
                      <StatusBadge
                        status={job.promoted ? 'SUCCESS' : 'WARNING'}
                        label={job.promoted ? 'Promoted (vNew)' : 'Active Preserved'}
                      />
                    </td>
                    <td style={{ fontSize: '0.82rem', maxWidth: 400 }}>{job.decision_reason || 'In evaluation...'}</td>
                    <td style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>
                      {new Date(job.created_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
