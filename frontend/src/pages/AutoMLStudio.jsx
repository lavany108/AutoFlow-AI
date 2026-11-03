import React, { useState, useEffect, useRef } from 'react';
import {
  Cpu, Play, Trophy, Clock, TrendingUp, AlertCircle,
  CheckCircle, ShieldAlert, Target, BarChart2
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, Cell
} from 'recharts';
import { api } from '../api';

const POLL_MS = 2000;

function LeaderboardRow({ item, rank, onTune }) {
  const isTop = rank === 1;
  const pr = item.metrics?.pr_auc;
  const f2 = item.metrics?.f2;
  const rec = item.metrics?.recall ?? item.metrics?.detection_rate;

  return (
    <tr style={{ background: isTop ? 'rgba(239,68,68,0.05)' : undefined }}>
      <td>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {isTop && <Trophy size={13} style={{ color: 'var(--yellow-400)' }} />}
          <span style={{ color: 'var(--text-primary)', fontWeight: isTop ? 700 : 500 }}>
            #{rank}
          </span>
        </div>
      </td>
      <td style={{ fontWeight: 600, color: isTop ? 'var(--red-400)' : 'var(--text-primary)' }}>
        {item.algorithm}
      </td>
      <td>
        <span className={`badge ${item.status === 'SUCCESS' ? 'badge-green' : 'badge-red'}`}>
          {item.status}
        </span>
      </td>
      <td className="mono" style={{ color: 'var(--purple-400)', fontWeight: 700 }}>
        {pr != null ? pr.toFixed(4) : '—'}
      </td>
      <td className="mono" style={{ color: 'var(--orange-400)' }}>
        {f2 != null ? f2.toFixed(4) : '—'}
      </td>
      <td className="mono" style={{ color: 'var(--cyan-400)' }}>
        {rec != null ? `${(rec * 100).toFixed(1)}%` : '—'}
      </td>
      <td className="mono" style={{ color: 'var(--text-muted)' }}>
        {item.cv_mean?.toFixed(4) ?? '—'} ± {item.cv_std?.toFixed(3) ?? '—'}
      </td>
      <td className="mono" style={{ color: 'var(--text-muted)' }}>
        {item.fit_time_seconds?.toFixed(1)}s
      </td>
      {onTune && item.model_id && (
        <td>
          <button className="btn btn-secondary btn-sm" onClick={() => onTune(item.model_id)}>
            <Target size={11} /> Tune
          </button>
        </td>
      )}
    </tr>
  );
}

export default function AutoMLStudio({
  currentDataset,
  activeModel,
  onModelTrained,
  onNavigateToRegistry,
}) {
  const [launching, setLaunching] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const [tuning, setTuning] = useState(false);
  const [tuneModelId, setTuneModelId] = useState(null);
  const [useSMOTE, setUseSMOTE] = useState(true);
  const [cvSplits, setCvSplits] = useState(5);
  const pollerRef = useRef(null);

  // Poll job status
  useEffect(() => {
    if (!jobId) return;
    pollerRef.current = setInterval(async () => {
      try {
        const j = await api.getJob(jobId);
        setJob(j);
        if (['COMPLETED', 'FAILED'].includes(j.status)) {
          clearInterval(pollerRef.current);
          if (j.status === 'COMPLETED') onModelTrained();
        }
      } catch (_) {}
    }, POLL_MS);
    return () => clearInterval(pollerRef.current);
  }, [jobId, onModelTrained]);

  const startTraining = async () => {
    if (!currentDataset?.target_column) {
      setError('Please configure a target column first (in Data Ingestion).');
      return;
    }
    setLaunching(true);
    setError(null);
    setJob(null);
    try {
      const j = await api.startBenchmark({
        dataset_id: currentDataset.id,
        test_size: 0.2,
        cv_splits: cvSplits,
        use_smote: useSMOTE,
      });
      setJobId(j.job_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setLaunching(false);
    }
  };

  const startTuning = async (modelId) => {
    setTuning(true);
    setTuneModelId(modelId);
    try {
      await api.startTuning(modelId, 15, 120);
      onModelTrained();
    } catch (err) {
      alert(`Tuning error: ${err.message}`);
    } finally {
      setTuning(false);
      setTuneModelId(null);
    }
  };

  const leaderboard = job?.leaderboard || [];
  const isRunning = job && ['PENDING', 'RUNNING'].includes(job.status);
  const isDone = job?.status === 'COMPLETED';

  // Chart data for comparison
  const chartData = leaderboard.filter(i => i.status === 'SUCCESS').map(i => ({
    name: i.algorithm.split(' ').slice(0, 2).join('\n'),
    'PR-AUC': +(i.metrics?.pr_auc ?? 0).toFixed(4),
    'F2':     +(i.metrics?.f2 ?? 0).toFixed(4),
    'Recall': +(i.metrics?.recall ?? i.metrics?.detection_rate ?? 0).toFixed(4),
  }));

  return (
    <div className="page-content fade-in">
      {/* Config card */}
      <div className="card section-gap">
        <div className="card-header">
          <div>
            <div className="card-title">
              <Cpu size={14} style={{ color: 'var(--red-400)' }} />
              Fraud Detection Training
            </div>
            <div className="card-subtitle">
              Benchmarks 5 class-weight-aware classifiers with fraud-optimized metrics (PR-AUC, F2)
            </div>
          </div>
          <button
            className="btn btn-primary"
            onClick={startTraining}
            disabled={launching || isRunning || !currentDataset?.target_column}
          >
            <Play size={14} />
            {launching ? 'Launching…' : isRunning ? 'Training…' : 'Start Benchmark'}
          </button>
        </div>

        {!currentDataset ? (
          <div className="alert-banner warning">
            <AlertCircle size={14} />
            Upload and configure a dataset first.
          </div>
        ) : !currentDataset.target_column ? (
          <div className="alert-banner warning">
            <AlertCircle size={14} />
            Configure target column in <strong>Data Ingestion</strong> before training.
          </div>
        ) : (
          <div style={{ display: 'flex', gap: 24, alignItems: 'center', flexWrap: 'wrap' }}>
            <div>
              <div className="form-label">Dataset</div>
              <div style={{ fontSize: 13, color: 'var(--text-primary)' }}>{currentDataset.original_filename}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{currentDataset.row_count?.toLocaleString()} rows · Target: {currentDataset.target_column}</div>
            </div>
            <div>
              <div className="form-label">CV Folds</div>
              <select
                className="form-select"
                value={cvSplits}
                onChange={(e) => setCvSplits(Number(e.target.value))}
                style={{ width: 80 }}
              >
                {[3, 5, 8].map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </div>
            <div>
              <div className="form-label">SMOTE Resampling</div>
              <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={useSMOTE}
                  onChange={(e) => setUseSMOTE(e.target.checked)}
                  style={{ accentColor: 'var(--red-500)' }}
                />
                <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                  {useSMOTE ? 'Enabled (minority oversampling)' : 'Disabled'}
                </span>
              </label>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', background: 'var(--bg-elevated)', padding: '6px 12px', borderRadius: 6 }}>
              Primary metric: <strong style={{ color: 'var(--purple-400)' }}>PR-AUC</strong> · Candidates: Logistic, RF, ET, HGB, GB
            </div>
          </div>
        )}

        {error && (
          <div className="alert-banner danger" style={{ marginTop: 14 }}>
            <AlertCircle size={14} /> {error}
          </div>
        )}
      </div>

      {/* Job progress */}
      {job && (
        <div className="card section-gap">
          <div className="card-header">
            <div className="card-title">
              {isRunning ? <Cpu size={14} className="spin" style={{ color: 'var(--orange-400)' }} /> : <CheckCircle size={14} style={{ color: 'var(--green-400)' }} />}
              {isRunning ? 'Training in Progress' : `Training ${job.status}`}
            </div>
            {isDone && (
              <button className="btn btn-success btn-sm" onClick={onNavigateToRegistry}>
                View Registry <CheckCircle size={12} />
              </button>
            )}
          </div>
          <div style={{ marginBottom: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 6 }}>
              <span style={{ color: 'var(--text-secondary)' }}>{job.current_step}</span>
              <span className="mono" style={{ color: 'var(--text-muted)' }}>{job.progress?.toFixed(0)}%</span>
            </div>
            <div className="progress-bar-container">
              <div className="progress-bar-fill" style={{ width: `${job.progress}%` }} />
            </div>
          </div>

          {/* Leaderboard table */}
          {leaderboard.length > 0 && (
            <>
              <div className="card-title" style={{ marginBottom: 10 }}>
                <Trophy size={14} style={{ color: 'var(--yellow-400)' }} />
                Model Leaderboard
              </div>
              <div className="data-table-container" style={{ marginBottom: 16 }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Algorithm</th>
                      <th>Status</th>
                      <th style={{ color: 'var(--purple-400)' }}>PR-AUC ↑</th>
                      <th style={{ color: 'var(--orange-400)' }}>F2 ↑</th>
                      <th style={{ color: 'var(--cyan-400)' }}>Recall ↑</th>
                      <th>CV (mean ± std)</th>
                      <th>Fit Time</th>
                      {isDone && <th>Actions</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {leaderboard.map((item, idx) => (
                      <LeaderboardRow
                        key={item.algorithm}
                        item={item}
                        rank={idx + 1}
                        onTune={isDone ? startTuning : null}
                      />
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Metric comparison chart */}
              {chartData.length > 0 && (
                <div>
                  <div className="card-title" style={{ marginBottom: 10 }}>
                    <BarChart2 size={14} style={{ color: 'var(--blue-400)' }} />
                    Metric Comparison
                  </div>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={chartData} margin={{ top: 0, right: 10, left: -20, bottom: 10 }}>
                      <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                      <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                      <Tooltip
                        contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', fontSize: 12 }}
                      />
                      <Bar dataKey="PR-AUC" fill="var(--purple-400)" radius={[3, 3, 0, 0]} />
                      <Bar dataKey="F2" fill="var(--orange-400)" radius={[3, 3, 0, 0]} />
                      <Bar dataKey="Recall" fill="var(--cyan-400)" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Tuning status */}
      {tuning && (
        <div className="alert-banner info">
          <Target size={14} className="spin" />
          Hyperparameter tuning in progress for model {tuneModelId}…
        </div>
      )}

      {/* Info cards */}
      {!job && (
        <div className="grid-3">
          {[
            { icon: ShieldAlert, color: 'var(--red-400)', title: 'Class-Weighted Models', desc: 'All 5 candidates use class_weight="balanced" to handle the ~4.5% fraud rate imbalance.' },
            { icon: TrendingUp, color: 'var(--purple-400)', title: 'PR-AUC Primary Metric', desc: 'Precision-Recall AUC outperforms ROC-AUC for imbalanced fraud datasets by penalizing false positives.' },
            { icon: Target, color: 'var(--orange-400)', title: 'F2 Score + SMOTE', desc: 'F2 score weights recall 2× over precision. SMOTE generates synthetic minority samples to balance training data.' },
          ].map(({ icon: Icon, color, title, desc }) => (
            <div key={title} className="card" style={{ borderColor: `${color}22` }}>
              <div style={{ display: 'flex', gap: 10, marginBottom: 8 }}>
                <Icon size={18} style={{ color, flexShrink: 0, marginTop: 2 }} />
                <div style={{ fontWeight: 700, fontSize: 13 }}>{title}</div>
              </div>
              <p style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.6 }}>{desc}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
