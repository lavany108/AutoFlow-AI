import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

n_samples = 1200

# 1. Customer Churn Dataset (Classification)
genders = np.random.choice(['Male', 'Female'], size=n_samples)
senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16])
partner = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.48, 0.52])
dependents = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.30, 0.70])
tenure = np.random.randint(1, 72, size=n_samples)
phone_service = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.90, 0.10])
multiple_lines = np.where(phone_service == 'No', 'No phone service', np.random.choice(['Yes', 'No'], size=n_samples, p=[0.42, 0.58]))
internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], size=n_samples, p=[0.34, 0.44, 0.22])
online_security = np.where(internet_service == 'No', 'No internet service', np.random.choice(['Yes', 'No'], size=n_samples, p=[0.29, 0.71]))
online_backup = np.where(internet_service == 'No', 'No internet service', np.random.choice(['Yes', 'No'], size=n_samples, p=[0.34, 0.66]))
device_protection = np.where(internet_service == 'No', 'No internet service', np.random.choice(['Yes', 'No'], size=n_samples, p=[0.34, 0.66]))
tech_support = np.where(internet_service == 'No', 'No internet service', np.random.choice(['Yes', 'No'], size=n_samples, p=[0.29, 0.71]))
contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples, p=[0.55, 0.21, 0.24])
paperless_billing = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.59, 0.41])
payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], size=n_samples)

# Monthly charges depend on services
base_charge = 20.0
monthly_charges = (
    base_charge
    + (phone_service == 'Yes') * 20.0
    + (multiple_lines == 'Yes') * 10.0
    + (internet_service == 'Fiber optic') * 45.0
    + (internet_service == 'DSL') * 25.0
    + (online_security == 'Yes') * 5.0
    + (tech_support == 'Yes') * 5.0
    + np.random.normal(0, 3, size=n_samples)
).round(2)
monthly_charges = np.clip(monthly_charges, 18.0, 118.0)

total_charges = (monthly_charges * tenure + np.random.normal(0, 10, size=n_samples)).round(2)
total_charges = np.maximum(total_charges, monthly_charges)

# Churn probability based on contract, tenure, fiber optic, tech support
churn_logit = (
    -1.5
    + (contract == 'Month-to-month') * 1.4
    - (contract == 'Two year') * 1.8
    - (tenure / 30.0)
    + (internet_service == 'Fiber optic') * 0.8
    - (tech_support == 'Yes') * 0.7
    + (payment_method == 'Electronic check') * 0.5
    + (senior_citizen == 1) * 0.3
)
churn_prob = 1 / (1 + np.exp(-churn_logit))
churn = np.where(np.random.rand(n_samples) < churn_prob, 'Yes', 'No')

# Inject a few missing values in total_charges and tech_support to test imputation
total_charges = total_charges.astype(object)
nan_idx_1 = np.random.choice(n_samples, size=15, replace=False)
total_charges[nan_idx_1] = np.nan

df_churn = pd.DataFrame({
    'tenure': tenure,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
    'senior_citizen': senior_citizen,
    'partner': partner,
    'dependents': dependents,
    'phone_service': phone_service,
    'multiple_lines': multiple_lines,
    'internet_service': internet_service,
    'online_security': online_security,
    'online_backup': online_backup,
    'device_protection': device_protection,
    'tech_support': tech_support,
    'contract': contract,
    'paperless_billing': paperless_billing,
    'payment_method': payment_method,
    'churn': churn
})

df_churn.to_csv('/Users/lavanyasmacbookair/AutoFlow-AI/sample_data/customer_churn.csv', index=False)
print("Generated sample_data/customer_churn.csv successfully with shape:", df_churn.shape)

