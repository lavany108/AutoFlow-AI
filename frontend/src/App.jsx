import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';

import OverviewDashboard from './pages/OverviewDashboard';
import DatasetExplorer from './pages/DatasetExplorer';
import AutoMLStudio from './pages/AutoMLStudio';
import ModelRegistry from './pages/ModelRegistry';
import PredictionPlayground from './pages/PredictionPlayground';
import ThresholdOptimizer from './pages/ThresholdOptimizer';
import ExplainabilityPage from './pages/ExplainabilityPage';
import MonitoringCenter from './pages/MonitoringCenter';

import { api } from './api';

export default function App() {
  const [currentTab, setCurrentTab] = useState('overview');
  const [isBackendHealthy, setIsBackendHealthy] = useState(true);
  const [datasets, setDatasets] = useState([]);
  const [currentDataset, setCurrentDataset] = useState(null);
  const [models, setModels] = useState([]);
  const [activeModel, setActiveModel] = useState(null);
  const [monitoringSummary, setMonitoringSummary] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    refreshAllData();
  }, []);

  const refreshAllData = async () => {
    setLoading(true);
    try {
      try { await api.getHealth(); setIsBackendHealthy(true); }
      catch { setIsBackendHealthy(false); }

      const dsList = await api.getDatasets();
      setDatasets(dsList || []);
      if (dsList?.length > 0 && !currentDataset) {
        const fullDs = await api.getDataset(dsList[0].id);
        setCurrentDataset(fullDs);
      } else if (currentDataset?.id) {
        const fullDs = await api.getDataset(currentDataset.id).catch(() => currentDataset);
        setCurrentDataset(fullDs);
      }

      const mList = await api.getModels();
      setModels(mList || []);
      const active = (mList || []).find((m) => m.is_active);
      if (active) {
        const fullActive = await api.getModel(active.id);
        setActiveModel(fullActive);
      } else {
        setActiveModel(null);
      }

      try {
        const mon = await api.getMonitoringSummary();
        setMonitoringSummary(mon);
      } catch (_) {
        setMonitoringSummary(null);
      }
    } catch (err) {
      console.error('Error refreshing data:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDatasetUploaded = async (uploaded) => {
    const full = await api.getDataset(uploaded.id).catch(() => uploaded);
    setCurrentDataset(full);
    const dsList = await api.getDatasets();
    setDatasets(dsList || []);
  };

  const handleSelectDataset = async (ds) => {
    const full = await api.getDataset(ds.id);
    setCurrentDataset(full);
  };

  const handleModelTrained = async () => {
    await refreshAllData();
  };

  const handleModelActivated = async (activated) => {
    const full = await api.getModel(activated.id);
    setActiveModel(full);
    const mList = await api.getModels();
    setModels(mList || []);
  };

  const handlePredictionLogged = async () => {
    try {
      const mon = await api.getMonitoringSummary();
      setMonitoringSummary(mon);
    } catch (_) {}
  };

  return (
    <div className="app-container">
      <Sidebar
        currentTab={currentTab}
        onSelectTab={setCurrentTab}
        isBackendHealthy={isBackendHealthy}
        activeModel={activeModel}
      />

      <div className="main-content">
        <Header
          currentTab={currentTab}
          activeModel={activeModel}
          currentDataset={currentDataset}
          onRefresh={refreshAllData}
          loading={loading}
        />

        {!isBackendHealthy && (
          <div style={{ padding: '10px 24px 0' }}>
            <div className="alert-banner danger">
              <strong>Backend Disconnected:</strong> Could not reach the FastAPI server at{' '}
              <code>http://localhost:8000</code>. Start it with:{' '}
              <code>cd /path/to/AutoFlow-AI && backend/.venv/bin/uvicorn backend.main:app --reload</code>
            </div>
          </div>
        )}

        {currentTab === 'overview' && (
          <OverviewDashboard
            datasets={datasets}
            models={models}
            activeModel={activeModel}
            monitoringSummary={monitoringSummary}
            onNavigate={setCurrentTab}
          />
        )}
        {currentTab === 'datasets' && (
          <DatasetExplorer
            datasets={datasets}
            currentDataset={currentDataset}
            onSelectDataset={handleSelectDataset}
            onDatasetUploaded={handleDatasetUploaded}
            onNavigateToStudio={() => setCurrentTab('studio')}
          />
        )}
        {currentTab === 'studio' && (
          <AutoMLStudio
            currentDataset={currentDataset}
            activeModel={activeModel}
            onModelTrained={handleModelTrained}
            onNavigateToRegistry={() => setCurrentTab('registry')}
          />
        )}
        {currentTab === 'registry' && (
          <ModelRegistry
            models={models}
            activeModel={activeModel}
            onModelActivated={handleModelActivated}
            onNavigateToPredict={() => setCurrentTab('predict')}
          />
        )}
        {currentTab === 'predict' && (
          <PredictionPlayground
            activeModel={activeModel}
            currentDataset={currentDataset}
            onPredictionLogged={handlePredictionLogged}
          />
        )}
        {currentTab === 'threshold' && (
          <ThresholdOptimizer activeModel={activeModel} />
        )}
        {currentTab === 'explain' && (
          <ExplainabilityPage activeModel={activeModel} />
        )}
        {currentTab === 'monitor' && (
          <MonitoringCenter
            activeModel={activeModel}
            monitoringSummary={monitoringSummary}
            onRefreshMonitoring={handlePredictionLogged}
            onModelRetrained={refreshAllData}
          />
        )}
      </div>
    </div>
  );
}
