/**
 * AutoFlow AI — Fraud Detection Platform
 * Centralized API client for all backend interactions
 */

const BASE_URL = "/api";

async function request(path, options = {}) {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const data = await res.json();
      detail = data.detail || JSON.stringify(data);
    } catch (_) {}
    throw new Error(detail);
  }
  if (res.status === 204) return null;
  return res.json();
}

// ─── System ──────────────────────────────────────────────────────────────────
const getHealth = () => request("/health");

// ─── Datasets ────────────────────────────────────────────────────────────────
const uploadDataset = (file) => {
  const form = new FormData();
  form.append("file", file);
  return fetch(`${BASE_URL}/datasets/upload`, { method: "POST", body: form })
    .then(async (res) => {
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || `HTTP ${res.status}`);
      }
      return res.json();
    });
};

const getDatasets = () => request("/datasets");
const getDataset = (id) => request(`/datasets/${id}`);
const getDatasetPreview = (id, limit = 50) =>
  request(`/datasets/${id}/preview?limit=${limit}`);
const configureTarget = (id, target_column, task_type = null) =>
  request(`/datasets/${id}/configure-target`, {
    method: "POST",
    body: JSON.stringify({ target_column, task_type }),
  });
const deleteDataset = (id) => request(`/datasets/${id}`, { method: "DELETE" });

// ─── Training ────────────────────────────────────────────────────────────────
const startBenchmark = (payload) =>
  request("/training/benchmark", {
    method: "POST",
    body: JSON.stringify(payload),
  });
const getJob = (jobId) => request(`/training/jobs/${jobId}`);
const listJobs = () => request("/training/jobs");

// ─── Tuning ──────────────────────────────────────────────────────────────────
const startTuning = (model_id, n_trials = 10, timeout_seconds = 180) =>
  request("/training/tune", {
    method: "POST",
    body: JSON.stringify({ model_id, n_trials, timeout_seconds }),
  });

// ─── Models ──────────────────────────────────────────────────────────────────
const getModels = () => request("/models");
const getModel = (id) => request(`/models/${id}`);
const activateModel = (id) =>
  request(`/models/${id}/activate`, { method: "POST" });
const deleteModel = (id) => request(`/models/${id}`, { method: "DELETE" });

// ─── Predictions ─────────────────────────────────────────────────────────────
const predictSingle = (features, model_id = null) =>
  request("/predictions/predict", {
    method: "POST",
    body: JSON.stringify({ features, model_id }),
  });
const predictBatch = (records, model_id = null) =>
  request("/predictions/batch", {
    method: "POST",
    body: JSON.stringify({ records, model_id }),
  });
const logActualOutcome = (prediction_log_id, actual_label) =>
  request("/predictions/actual-outcome", {
    method: "POST",
    body: JSON.stringify({ prediction_log_id, actual_label }),
  });

// ─── Fraud-Specific ──────────────────────────────────────────────────────────
const scoreTransaction = (transaction, model_id = null, threshold = null) =>
  request("/fraud/score-transaction", {
    method: "POST",
    body: JSON.stringify({ transaction, model_id, threshold }),
  });

const optimizeThreshold = (payload) =>
  request("/fraud/optimize-threshold", {
    method: "POST",
    body: JSON.stringify(payload),
  });

const getBusinessImpact = (model_id, cost_fp = 10, cost_fn = 500, avg_fraud_amount = 500) =>
  request(
    `/fraud/business-impact/${model_id}?cost_fp=${cost_fp}&cost_fn=${cost_fn}&avg_fraud_amount=${avg_fraud_amount}`
  );

// ─── Explainability ──────────────────────────────────────────────────────────
const getExplainability = (model_id, sample_size = 150) =>
  request(`/explainability/${model_id}?sample_size=${sample_size}`);

// ─── Monitoring & Retraining ──────────────────────────────────────────────
const getMonitoringSummary = (model_id = null) => {
  const qs = model_id ? `?model_id=${model_id}` : "";
  return request(`/monitoring/summary${qs}`);
};

const triggerRetraining = (modelIdOrPayload, reason = "Manual UI Trigger", performance_tolerance = 0.05) => {
  const payload =
    typeof modelIdOrPayload === "object"
      ? modelIdOrPayload
      : { model_id: modelIdOrPayload, reason, performance_tolerance };
  return request("/retraining/trigger", {
    method: "POST",
    body: JSON.stringify(payload),
  });
};
const triggerRetrain = triggerRetraining;

const getRetrainJobs = () => request("/retraining/jobs");
const getRetrainJob = (jobId) => request(`/retraining/jobs/${jobId}`);
const submitActualOutcome = logActualOutcome;

export const api = {
  getHealth,
  uploadDataset,
  getDatasets,
  getDataset,
  getDatasetPreview,
  configureTarget,
  deleteDataset,
  startBenchmark,
  getJob,
  listJobs,
  startTuning,
  getModels,
  getModel,
  activateModel,
  deleteModel,
  predictSingle,
  predictBatch,
  submitActualOutcome,
  logActualOutcome,
  scoreTransaction,
  optimizeThreshold,
  getBusinessImpact,
  getExplainability,
  getMonitoringSummary,
  triggerRetrain,
  triggerRetraining,
  getRetrainJobs,
  getRetrainJob,
};
