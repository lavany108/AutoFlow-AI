import React from 'react';

export default function StatusBadge({ status, label }) {
  const normStatus = (status || '').toUpperCase();
  let type = 'info';

  if (['SUCCESS', 'COMPLETED', 'ACTIVE', 'HEALTHY', 'STABLE'].includes(normStatus)) {
    type = 'success';
  } else if (['RUNNING', 'PENDING', 'WARNING', 'UNKNOWN'].includes(normStatus)) {
    type = 'warning';
  } else if (['FAILED', 'CRITICAL', 'DRIFT DETECTED', 'DEGRADED', 'ERROR'].includes(normStatus)) {
    type = 'danger';
  }

  return (
    <span className={`badge badge-${type}`}>
      {label || status}
    </span>
  );
}
