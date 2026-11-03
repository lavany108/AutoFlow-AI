import React from 'react';
import { RefreshCw, Shield, AlertTriangle } from 'lucide-react';

const PAGE_TITLES = {
  overview:   'Command Center',
  datasets:   'Data Ingestion & Profiling',
  studio:     'Model Training & Benchmarking',
  registry:   'Model Registry & Experiments',
  predict:    'Transaction Scoring Engine',
  threshold:  'Decision Threshold Optimizer',
  explain:    'Model Explainability (SHAP)',
  monitor:    'Drift Monitor & Alerts',
};

export default function Header({ currentTab, activeModel, currentDataset, onRefresh, loading }) {
  return (
    <div className="header">
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <div className="header-title">{PAGE_TITLES[currentTab] || 'AutoFlow AI'}</div>
      </div>
      <div className="header-right">
        {currentDataset && (
          <div className="header-badge">
            <Shield size={11} style={{ color: 'var(--blue-400)' }} />
            {currentDataset.original_filename || currentDataset.filename}
          </div>
        )}
        {activeModel && (
          <div className="header-badge" style={{ borderColor: 'rgba(239,68,68,0.2)', color: 'var(--red-400)' }}>
            <AlertTriangle size={11} />
            {activeModel.algorithm?.split(' ').slice(0, 2).join(' ')} v{activeModel.version}
          </div>
        )}
        <button
          className={`btn btn-ghost btn-icon ${loading ? 'spin' : ''}`}
          onClick={onRefresh}
          disabled={loading}
          title="Refresh data"
        >
          <RefreshCw size={15} />
        </button>
      </div>
    </div>
  );
}
