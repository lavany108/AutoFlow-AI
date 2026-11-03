import React from 'react';

export default function MetricCard({ title, value, subtitle, icon: Icon, badge, color = 'indigo' }) {
  return (
    <div className="metric-card">
      <div className="metric-card-top">
        <span>{title}</span>
        {Icon && (
          <div className="metric-card-icon" style={{ color: `var(--${color}, #4f46e5)` }}>
            <Icon size={18} />
          </div>
        )}
      </div>
      <div className="metric-card-value">{value}</div>
      <div className="metric-card-footer" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>{subtitle}</span>
        {badge}
      </div>
    </div>
  );
}
