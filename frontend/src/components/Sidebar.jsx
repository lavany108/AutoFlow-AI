import React from 'react';
import {
  LayoutDashboard,
  Database,
  Cpu,
  Package,
  Zap,
  Brain,
  Activity,
  ShieldAlert,
  Sliders,
} from 'lucide-react';

const NAV_ITEMS = [
  { id: 'overview',   label: 'Command Center',     icon: LayoutDashboard, section: 'PLATFORM' },
  { id: 'datasets',   label: 'Data Ingestion',      icon: Database,        section: 'PLATFORM' },
  { id: 'studio',     label: 'Model Training',      icon: Cpu,             section: 'MODELS' },
  { id: 'registry',   label: 'Model Registry',      icon: Package,         section: 'MODELS' },
  { id: 'predict',    label: 'Transaction Scoring', icon: Zap,             section: 'DETECTION' },
  { id: 'threshold',  label: 'Threshold Optimizer', icon: Sliders,         section: 'DETECTION' },
  { id: 'explain',    label: 'Explainability',      icon: Brain,           section: 'DETECTION' },
  { id: 'monitor',    label: 'Drift Monitor',       icon: Activity,        section: 'OPS' },
];

export default function Sidebar({ currentTab, onSelectTab, isBackendHealthy, activeModel }) {
  let sectionSeen = new Set();

  return (
    <div className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">
          <ShieldAlert size={18} color="#fff" />
        </div>
        <div className="sidebar-logo-text">
          <h1>AutoFlow AI</h1>
          <span>Fraud Intelligence</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => {
          const showSection = !sectionSeen.has(item.section);
          if (showSection) sectionSeen.add(item.section);
          return (
            <React.Fragment key={item.id}>
              {showSection && (
                <div className="nav-section-label" style={{ marginTop: showSection && item.id !== 'overview' ? 10 : 0 }}>
                  {item.section}
                </div>
              )}
              <div
                className={`nav-item ${currentTab === item.id ? 'active' : ''}`}
                onClick={() => onSelectTab(item.id)}
              >
                <span className="nav-item-icon">
                  <item.icon size={15} />
                </span>
                {item.label}
              </div>
            </React.Fragment>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="sidebar-footer">
        <div className={`status-pill ${isBackendHealthy ? 'healthy' : 'unhealthy'}`}>
          <span className={`status-dot ${isBackendHealthy ? 'green' : 'red'}`} />
          {isBackendHealthy ? 'Backend Online' : 'Backend Offline'}
        </div>
        {activeModel && (
          <div className="status-pill" style={{ background: 'rgba(96,165,250,0.08)', color: 'var(--blue-400)', fontSize: 11, wordBreak: 'break-all' }}>
            <Package size={11} style={{ flexShrink: 0 }} />
            Active: {activeModel.name?.split(' ').slice(0, 2).join(' ')}
          </div>
        )}
      </div>
    </div>
  );
}
