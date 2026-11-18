import os
import random
from typing import Tuple
import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def generate_transaction_network(
    num_accounts: int = 4000,
    num_transactions: int = 28000,
    fraud_ring_count: int = 35,
    ring_size_min: int = 4,
    ring_size_max: int = 7,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates realistic account profiles and directed financial transactions
    where individual node stats overlap heavily between legitimate high-velocity
    users and fraud rings, making multi-hop graph topology essential for detection.
    """
    set_seed(seed)

    # 1. Generate Account Nodes
    account_ids = np.arange(num_accounts)
    account_age_days = np.random.exponential(scale=450, size=num_accounts).clip(5, 3650)
    kyc_verified = np.random.binomial(1, 0.88, size=num_accounts)
    avg_monthly_balance = np.random.lognormal(mean=9.5, sigma=1.1, size=num_accounts)
    device_cluster_id = np.random.randint(0, num_accounts // 2, size=num_accounts)

    accounts_df = pd.DataFrame({
        "account_id": account_ids,
        "account_age_days": account_age_days,
        "kyc_verified": kyc_verified,
        "avg_monthly_balance": avg_monthly_balance,
        "device_cluster_id": device_cluster_id,
        "is_fraud_node": np.zeros(num_accounts, dtype=int),
    })

    # 2. Normal Background Transactions (including legitimate high-velocity burst users)
    weights = np.random.pareto(a=1.5, size=num_accounts) + 1.0
    probs = weights / weights.sum()

    src_nodes = np.random.choice(account_ids, size=num_transactions, p=probs)
    dst_nodes = np.random.choice(account_ids, size=num_transactions, p=probs)
    mask = src_nodes != dst_nodes
    src_nodes, dst_nodes = src_nodes[mask], dst_nodes[mask]

    n_normal = len(src_nodes)
    timestamps = np.sort(np.random.uniform(0, 30 * 24 * 3600, size=n_normal))
    amounts = np.random.lognormal(mean=7.2, sigma=1.3, size=n_normal).clip(10.0, 250000.0)
    channel = np.random.choice([0, 1, 2], size=n_normal, p=[0.65, 0.25, 0.10])

    tx_records = pd.DataFrame({
        "src": src_nodes,
        "dst": dst_nodes,
        "amount": amounts,
        "timestamp": timestamps,
        "channel": channel,
        "is_fraud": np.zeros(n_normal, dtype=int),
    })

    # Inject legitimate burst/pass-through activity so single-node velocity isn't trivially separable
    legit_burst_accounts = np.random.choice(account_ids, size=140, replace=False)
    legit_burst_tx = []
    for acc_id in legit_burst_accounts:
        t0 = random.uniform(1 * 24 * 3600, 29 * 24 * 3600)
        peers = np.random.choice(account_ids, size=4, replace=False)
        for k, peer in enumerate(peers):
            if peer == acc_id:
                continue
            legit_burst_tx.append({
                "src": int(acc_id) if k % 2 == 0 else int(peer),
                "dst": int(peer) if k % 2 == 0 else int(acc_id),
                "amount": float(np.random.lognormal(mean=8.2, sigma=0.7)),
                "timestamp": t0 + k * random.uniform(180, 1800),
                "channel": 0,
                "is_fraud": 0,
            })

    # 3. Inject Collusion Rings & Layering Chains (Camouflaged individual stats, strong 2-hop cycle)
    available_pool = set(account_ids.tolist()) - set(legit_burst_accounts.tolist())
    fraud_tx_list = []

    for _ in range(fraud_ring_count):
        ring_size = random.randint(ring_size_min, ring_size_max)
        if len(available_pool) < ring_size:
            break

        ring_nodes = random.sample(list(available_pool), ring_size)
        available_pool -= set(ring_nodes)
        accounts_df.loc[ring_nodes, "is_fraud_node"] = 1

        # Camouflage individual account features to match normal distribution
        accounts_df.loc[ring_nodes, "kyc_verified"] = np.random.binomial(1, 0.85, size=ring_size)

        base_time = random.uniform(2 * 24 * 3600, 28 * 24 * 3600)
        base_amt = float(np.random.lognormal(mean=8.3, sigma=0.6))

        for hop in range(ring_size * 2):
            u = ring_nodes[hop % ring_size]
            v = ring_nodes[(hop + 1) % ring_size]
            hop_time = base_time + hop * random.uniform(300, 3600)
            hop_amt = base_amt * (0.97 ** (hop % ring_size)) * random.uniform(0.95, 1.05)

            fraud_tx_list.append({
                "src": u,
                "dst": v,
                "amount": hop_amt,
                "timestamp": hop_time,
                "channel": random.choice([0, 1]),
                "is_fraud": 1,
            })

    full_tx_df = (
        pd.concat([tx_records, pd.DataFrame(legit_burst_tx), pd.DataFrame(fraud_tx_list)], ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return accounts_df, full_tx_df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    acc, tx = generate_transaction_network()
    acc.to_csv("data/raw/accounts.csv", index=False)
    tx.to_csv("data/raw/transactions.csv", index=False)
    print(f"Generated {len(acc)} accounts ({acc['is_fraud_node'].sum()} fraud) and {len(tx)} transactions.")