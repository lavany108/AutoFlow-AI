import React from 'react';
import { Check, ArrowRight } from 'lucide-react';

const STAGES = [
  { id: 'datasets', label: '1. Upload CSV' },
  { id: 'datasets', label: '2. Profile & Target' },
  { id: 'studio', label: '3. Preprocess' },
  { id: 'studio', label: '4. Benchmark 5 Models' },
  { id: 'studio', label: '5. Optuna Tuning' },
  { id: 'registry', label: '6. Model Registry' },
  { id: 'predict', label: '7. Inference API' },
  { id: 'explain', label: '8. SHAP Explain' },
  { id: 'monitor', label: '9. Drift & Retrain' },
];

export default function PipelineStepper({ currentTab, onSelectTab, activeModel, currentDataset }) {
  const getStageStatus = (idx) => {
    if (idx <= 1 && currentDataset) return 'completed';
    if (idx >= 2 && idx <= 5 && activeModel) return 'completed';
    if (idx === 6 && activeModel) return 'completed';
    return '';
  };

  return (
    <div className="pipeline-stepper">
      {STAGES.map((st, idx) => {
        const isCompleted = getStageStatus(idx) === 'completed';
        const isCurrent = currentTab === st.id;

        return (
          <React.Fragment key={idx}>
            <div
              className={`step-item ${isCompleted ? 'completed' : ''} ${isCurrent ? 'active' : ''}`}
              onClick={() => onSelectTab(st.id)}
              title={`Jump to ${st.label}`}
            >
              <div className="step-number">
                {isCompleted ? <Check size={14} /> : idx + 1}
              </div>
              <span>{st.label}</span>
            </div>
            {idx < STAGES.length - 1 && (
              <div className={`step-divider ${isCompleted ? 'filled' : ''}`} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}
