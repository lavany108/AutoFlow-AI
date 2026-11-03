import React, { useState, useCallback } from 'react';
import {
  Upload, Database, ChevronRight, Info, CheckCircle,
  AlertTriangle, BarChart2, List, Trash2
} from 'lucide-react';
import { api } from '../api';

function ImbalanceBar({ ratio }) {
  const pct = (ratio * 100).toFixed(1);
  const color = ratio < 0.01 ? 'var(--risk-critical)' :
                ratio < 0.05 ? 'var(--risk-high)' :
                ratio < 0.15 ? 'var(--risk-medium)' : 'var(--risk-low)';
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4 }}>
        <span style={{ color: 'var(--text-muted)' }}>Fraud Ratio</span>
        <span style={{ color, fontWeight: 700 }}>{pct}%</span>
      </div>
      <div className="progress-bar-container">
        <div
          className="progress-bar-fill"
          style={{ width: `${Math.min(pct * 3, 100)}%`, background: `linear-gradient(90deg, ${color}, ${color}aa)` }}
        />
      </div>
    </div>
  );
}

export default function DatasetExplorer({
  datasets,
  currentDataset,
  onSelectDataset,
  onDatasetUploaded,
  onNavigateToStudio,
}) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [configuring, setConfiguring] = useState(false);
  const [targetCol, setTargetCol] = useState('');
  const [activeTab, setActiveTab] = useState('profile');

  const handleFileUpload = useCallback(async (file) => {
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    try {
      const res = await api.uploadDataset(file);
      await onDatasetUploaded(res);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  }, [onDatasetUploaded]);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  };

  const handleConfigure = async () => {
    if (!currentDataset || !targetCol) return;
    setConfiguring(true);
    try {
      const updated = await api.configureTarget(currentDataset.id, targetCol, 'classification');
      await onDatasetUploaded(updated);
    } catch (err) {
      alert(`Configure error: ${err.message}`);
    } finally {
      setConfiguring(false);
    }
  };

  const colInfo = currentDataset?.columns_info || {};
  const summaryStats = currentDataset?.summary_stats || {};
  const imbalance = currentDataset?.imbalance_stats;

  return (
    <div className="page-content fade-in">
      <div className="grid-2-1">
        {/* Main panel */}
        <div>
          {/* Upload zone */}
          {!currentDataset && (
            <div
              className={`upload-zone card ${dragging ? 'drag-over' : ''}`}
              style={{ marginBottom: 18 }}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => document.getElementById('file-input').click()}
            >
              <Upload size={32} style={{ color: 'var(--red-400)', marginBottom: 12 }} />
              <h3>Upload Transaction CSV</h3>
              <p style={{ marginTop: 6 }}>Drag & drop or click to select your fraud transaction dataset</p>
              <p style={{ marginTop: 4, fontSize: 11 }}>Supported: CSV · UTF-8</p>
              {uploading && <p style={{ marginTop: 10, color: 'var(--orange-400)', fontSize: 12 }}>Uploading…</p>}
              {uploadError && <p style={{ marginTop: 10, color: 'var(--red-400)', fontSize: 12 }}>{uploadError}</p>}
              <input
                id="file-input"
                type="file"
                accept=".csv"
                hidden
                onChange={(e) => handleFileUpload(e.target.files[0])}
              />
            </div>
          )}

          {currentDataset && (
            <>
              {/* Dataset info */}
              <div className="card section-gap">
                <div className="card-header">
                  <div>
                    <div className="card-title">
                      <Database size={14} style={{ color: 'var(--blue-400)' }} />
                      {currentDataset.original_filename}
                    </div>
                    <div className="card-subtitle">
                      {currentDataset.row_count?.toLocaleString()} rows · {currentDataset.column_count} features ·{' '}
                      {(currentDataset.file_size_bytes / 1024).toFixed(1)} KB
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    {currentDataset.target_column && (
                      <span className="badge badge-green">
                        <CheckCircle size={10} /> Target: {currentDataset.target_column}
                      </span>
                    )}
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={onNavigateToStudio}
                      disabled={!currentDataset.target_column}
                    >
                      Train Models <ChevronRight size={12} />
                    </button>
                  </div>
                </div>

                {/* Target configuration */}
                {!currentDataset.target_column && (
                  <div className="alert-banner warning">
                    <AlertTriangle size={14} />
                    Configure the target column (e.g., <code>is_fraud</code>) before training.
                  </div>
                )}

                <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 14 }}>
                  <select
                    className="form-select"
                    value={targetCol || currentDataset.target_column || ''}
                    onChange={(e) => setTargetCol(e.target.value)}
                    style={{ maxWidth: 220 }}
                  >
                    <option value="">Select target column...</option>
                    {(currentDataset.potential_targets || []).map((col) => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={handleConfigure}
                    disabled={configuring || !targetCol}
                  >
                    {configuring ? 'Configuring…' : 'Set Target'}
                  </button>
                </div>

                {/* Tabs */}
                <div className="tabs" style={{ marginBottom: 16 }}>
                  {['profile', 'stats', 'preview'].map(tab => (
                    <button
                      key={tab}
                      className={`tab-btn ${activeTab === tab ? 'active' : ''}`}
                      onClick={() => setActiveTab(tab)}
                    >
                      {tab === 'profile' ? 'Column Profile' : tab === 'stats' ? 'Statistics' : 'Preview'}
                    </button>
                  ))}
                </div>

                {activeTab === 'profile' && (
                  <div className="data-table-container">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Column</th>
                          <th>Type</th>
                          <th>Missing</th>
                          <th>Unique</th>
                          <th>Sample Values</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.values(colInfo).map((col) => (
                          <tr key={col.name}>
                            <td style={{ fontWeight: 600, color: col.name === currentDataset.target_column ? 'var(--red-400)' : 'var(--text-primary)' }}>
                              {col.name}
                              {col.name === currentDataset.target_column && <span className="badge badge-red" style={{ marginLeft: 6, fontSize: 9 }}>TARGET</span>}
                            </td>
                            <td>
                              <span className={`badge ${col.inferred_type === 'numerical' ? 'badge-blue' : col.inferred_type === 'boolean' ? 'badge-purple' : 'badge-gray'}`}>
                                {col.inferred_type}
                              </span>
                            </td>
                            <td style={{ color: col.missing_pct > 10 ? 'var(--orange-400)' : 'var(--text-muted)' }}>
                              {col.missing_pct}%
                            </td>
                            <td>{col.unique_count}</td>
                            <td style={{ color: 'var(--text-muted)', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                              {col.sample_values?.slice(0, 3).join(', ')}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                {activeTab === 'stats' && (
                  <div className="data-table-container">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Column</th>
                          <th>Type</th>
                          <th>Count</th>
                          <th>Mean / Top</th>
                          <th>Std / Freq</th>
                          <th>Min</th>
                          <th>Max</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(summaryStats).map(([col, stat]) => (
                          <tr key={col}>
                            <td style={{ fontWeight: 600 }}>{col}</td>
                            <td><span className={`badge ${stat.type === 'numerical' ? 'badge-blue' : 'badge-gray'}`}>{stat.type}</span></td>
                            <td>{stat.count}</td>
                            <td>{stat.type === 'numerical' ? stat.mean?.toFixed(3) : stat.top}</td>
                            <td>{stat.type === 'numerical' ? stat.std?.toFixed(3) : stat.top_freq}</td>
                            <td>{stat.min?.toFixed(2) ?? '—'}</td>
                            <td>{stat.max?.toFixed(2) ?? '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                {activeTab === 'preview' && (
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', textAlign: 'center', padding: 24 }}>
                    <List size={24} style={{ opacity: 0.3, marginBottom: 8 }} />
                    <div>Preview available via API at <code>/api/datasets/{currentDataset.id}/preview</code></div>
                  </div>
                )}
              </div>

              {/* Re-upload */}
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => document.getElementById('file-input2').click()}
              >
                <Upload size={12} /> Upload Different File
              </button>
              <input
                id="file-input2"
                type="file"
                accept=".csv"
                hidden
                onChange={(e) => handleFileUpload(e.target.files[0])}
              />
            </>
          )}
        </div>

        {/* Sidebar: dataset list + imbalance stats */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {imbalance && (
            <div className="card">
              <div className="card-title" style={{ marginBottom: 12 }}>
                <BarChart2 size={14} style={{ color: 'var(--orange-400)' }} />
                Class Imbalance
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <ImbalanceBar ratio={imbalance.fraud_ratio || 0} />
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 12 }}>
                  <div style={{ background: 'var(--bg-elevated)', borderRadius: 6, padding: 10, textAlign: 'center' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: 10, marginBottom: 4 }}>LEGIT</div>
                    <div style={{ fontWeight: 700, color: 'var(--green-400)', fontSize: 18, fontFamily: 'JetBrains Mono' }}>
                      {imbalance.legit_count?.toLocaleString()}
                    </div>
                  </div>
                  <div style={{ background: 'var(--bg-elevated)', borderRadius: 6, padding: 10, textAlign: 'center' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: 10, marginBottom: 4 }}>FRAUD</div>
                    <div style={{ fontWeight: 700, color: 'var(--red-400)', fontSize: 18, fontFamily: 'JetBrains Mono' }}>
                      {imbalance.fraud_count?.toLocaleString()}
                    </div>
                  </div>
                </div>
                <div style={{ fontSize: 11, padding: '8px 10px', background: 'rgba(251,146,60,0.08)', borderRadius: 6, color: 'var(--orange-400)', border: '1px solid rgba(251,146,60,0.15)' }}>
                  <strong>Severity:</strong> {imbalance.imbalance_severity}
                  <br />
                  <strong>Strategy:</strong> {imbalance.recommended_strategy}
                </div>
              </div>
            </div>
          )}

          <div className="card">
            <div className="card-title" style={{ marginBottom: 12 }}>
              <Database size={14} style={{ color: 'var(--blue-400)' }} />
              Loaded Datasets
            </div>
            {datasets.length === 0 ? (
              <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>No datasets uploaded yet.</div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {datasets.map((ds) => (
                  <div
                    key={ds.id}
                    className={`nav-item ${currentDataset?.id === ds.id ? 'active' : ''}`}
                    style={{ padding: '8px 10px', fontSize: 12, borderRadius: 6 }}
                    onClick={() => onSelectDataset(ds)}
                  >
                    <Database size={12} />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {ds.original_filename || ds.filename}
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                        {ds.row_count?.toLocaleString()} rows
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="card">
            <div className="card-title" style={{ marginBottom: 10 }}>
              <Info size={14} style={{ color: 'var(--purple-400)' }} />
              Sample Dataset
            </div>
            <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 10 }}>
              Use the included realistic fraud dataset with 5,000 transactions (4.5% fraud rate):
            </p>
            <code style={{ fontSize: 11, color: 'var(--text-code)', background: 'var(--bg-elevated)', padding: '6px 10px', borderRadius: 4, display: 'block' }}>
              sample_data/fraud_transactions.csv
            </code>
          </div>
        </div>
      </div>
    </div>
  );
}
