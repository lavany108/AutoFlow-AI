"""
Generate a realistic financial transaction fraud dataset.
- ~5,000 transactions
- ~4.5% fraud rate (realistic imbalance)
- Features: transaction amounts, merchant categories, time patterns, velocity, etc.
"""
import numpy as np
import pandas as pd
import random
from pathlib import Path

np.random.seed(42)
random.seed(42)

N = 5000
FRAUD_RATE = 0.045

n_fraud = int(N * FRAUD_RATE)
n_legit = N - n_fraud

def make_transactions(n, is_fraud):
    rng = np.random.default_rng(seed=1 if is_fraud else 0)

    # Transaction amount — fraud tends to be higher or at round numbers
    if is_fraud:
        amounts = np.concatenate([
            rng.uniform(800, 4000, n // 3),          # high-value fraud
            rng.choice([100, 200, 500, 1000, 2000], n // 3),  # card testing rounds
            rng.uniform(10, 200, n - 2*(n // 3)),    # small fraud
        ])
    else:
        amounts = rng.lognormal(mean=4.0, sigma=1.2, size=n)  # typical spend ~$55
        amounts = np.clip(amounts, 1, 3000)

    amounts = np.round(amounts, 2)

    # Hour of transaction
    if is_fraud:
        # Fraud spikes at night (1-5am)
        hours = rng.choice(list(range(1, 6)) + list(range(22, 24)), size=n)
    else:
        hours = rng.choice(range(8, 23), size=n, p=np.ones(15)/15)

    # Day of week (0=Mon)
    days = rng.integers(0, 7, size=n)

    # Merchant category
    categories = ["grocery", "gas_station", "restaurant", "online_retail",
                  "atm_withdrawal", "travel", "entertainment", "utilities"]
    cat_probs_fraud = [0.05, 0.08, 0.05, 0.35, 0.25, 0.12, 0.07, 0.03]
    cat_probs_legit = [0.22, 0.12, 0.18, 0.20, 0.08, 0.08, 0.07, 0.05]
    merchant_category = rng.choice(categories, size=n,
                                    p=cat_probs_fraud if is_fraud else cat_probs_legit)

    # Country — fraud more likely from unusual countries
    countries = ["US", "GB", "DE", "FR", "CN", "NG", "RO", "BR", "CA", "AU"]
    home_country_prob = [0.30, 0.12, 0.10, 0.10, 0.12, 0.10, 0.06, 0.05, 0.03, 0.02] if is_fraud \
                     else [0.65, 0.09, 0.06, 0.05, 0.03, 0.01, 0.01, 0.04, 0.04, 0.02]
    country = rng.choice(countries, size=n, p=home_country_prob)
    is_foreign_transaction = (country != "US").astype(int)

    # Transaction velocity — transactions in last hour
    if is_fraud:
        velocity_1h = rng.integers(3, 15, size=n)
        velocity_24h = rng.integers(8, 40, size=n)
    else:
        velocity_1h = rng.integers(0, 4, size=n)
        velocity_24h = rng.integers(1, 12, size=n)

    # Account age in days
    if is_fraud:
        account_age_days = rng.integers(1, 90, size=n)
    else:
        account_age_days = rng.integers(30, 3000, size=n)

    # Distance from home (km)
    if is_fraud:
        distance_from_home = rng.uniform(500, 5000, size=n)
    else:
        distance_from_home = rng.lognormal(3, 1.5, size=n)
        distance_from_home = np.clip(distance_from_home, 0, 2000)

    # Days since last transaction
    if is_fraud:
        days_since_last_txn = rng.integers(0, 3, size=n)
    else:
        days_since_last_txn = rng.integers(0, 30, size=n)

    # Has chip (chip transactions are harder to clone)
    chip_used = rng.choice([0, 1], size=n, p=[0.7, 0.3] if is_fraud else [0.2, 0.8])

    # Online transaction
    is_online = (merchant_category == "online_retail").astype(int)

    # Card type
    card_types = ["credit", "debit", "prepaid"]
    card_type = rng.choice(card_types, size=n,
                            p=[0.35, 0.20, 0.45] if is_fraud else [0.50, 0.40, 0.10])

    # Credit limit utilization
    if is_fraud:
        credit_utilization = rng.uniform(0.75, 1.0, size=n)
    else:
        credit_utilization = rng.uniform(0.0, 0.60, size=n)

    # Previous fraud flag on account
    prev_fraud_flag = rng.choice([0, 1], size=n, p=[0.3, 0.7] if is_fraud else [0.97, 0.03])

    return pd.DataFrame({
        "amount": amounts,
        "hour_of_day": hours,
        "day_of_week": days,
        "merchant_category": merchant_category,
        "country": country,
        "is_foreign_transaction": is_foreign_transaction,
        "velocity_last_1h": velocity_1h,
        "velocity_last_24h": velocity_24h,
        "account_age_days": account_age_days,
        "distance_from_home_km": np.round(distance_from_home, 1),
        "days_since_last_transaction": days_since_last_txn,
        "chip_used": chip_used,
        "is_online_transaction": is_online,
        "card_type": card_type,
        "credit_utilization": np.round(credit_utilization, 3),
        "previous_fraud_flag": prev_fraud_flag,
        "is_fraud": 1 if is_fraud else 0,
    })

legit_df = make_transactions(n_legit, is_fraud=False)
fraud_df = make_transactions(n_fraud, is_fraud=True)

df = pd.concat([legit_df, fraud_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Add transaction_id
df.insert(0, "transaction_id", [f"TXN{i:06d}" for i in range(len(df))])

print(f"Total transactions: {len(df)}")
print(f"Fraud: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
print(f"Features: {list(df.columns)}")
print(df.describe())

out_path = Path(__file__).parent / "fraud_transactions.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
