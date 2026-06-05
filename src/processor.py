import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_cdnow(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["customer_id", "date", 
                                               "quantity", "price"])
    df = df[df["customer_id"] != "customer_id"].dropna()
    df["customer_id"] = df["customer_id"].astype(int)
    df["date"] = pd.to_datetime(df["date"])
    df["quantity"] = df["quantity"].astype(int)
    df["price"] = df["price"].astype(float)
    return df.sort_values(["customer_id", "date"]).reset_index(drop=True)

def build_rfm(df: pd.DataFrame, obs_end: pd.Timestamp = None) -> pd.DataFrame:
    if obs_end is None:
        obs_end = df["date"].max()

    rfm = (
        df.groupby("customer_id")
        .agg(
            first_purchase=("date", "min"),
            last_purchase=("date", "max"),
            frequency=("date", "count"),
            monetary=("price", "sum"),
            avg_quantity=("quantity", "mean"),
            total_quantity=("quantity", "sum"),
        )
        .reset_index()
    )

    rfm["recency_days"] = (obs_end - rfm["last_purchase"]).dt.days
    rfm["T_days"] = (obs_end - rfm["first_purchase"]).dt.days
    rfm["purchase_span"] = (rfm["last_purchase"] - rfm["first_purchase"]).dt.days
    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]
    rfm["repeat_frequency"] = rfm["frequency"] - 1
    rfm["recency_T"] = rfm["T_days"] - rfm["recency_days"]

    return rfm

def make_survival_labels(rfm: pd.DataFrame, df: pd.DataFrame, obs_end: pd.Timestamp = None):
    if obs_end is None:
        obs_end = df["date"].max()

    second_purchase = (
        df.sort_values(["customer_id", "date"])
        .groupby("customer_id", as_index=False)
        .nth(1)[["customer_id", "date"]]
        .rename(columns={"date": "second_purchase_date"})
    )

    rfm2 = rfm.merge(second_purchase, on="customer_id", how="left")
    rfm2["event"] = rfm2["second_purchase_date"].notna().astype(int)
    rfm2["duration"] = np.where(
        rfm2["event"] == 1,
        (rfm2["second_purchase_date"] - rfm2["first_purchase"]).dt.days,
        (obs_end - rfm2["first_purchase"]).dt.days,
    )
    rfm2["duration"] = rfm2["duration"].clip(lower=1)
    return rfm2

def get_feature_columns():
    return [
        "recency_days",
        "frequency",
        "monetary",
        "T_days",
        "avg_order_value",
        "avg_quantity",
        "purchase_span",
    ]

def prepare_ml_data(rfm_full: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    feat_cols = get_feature_columns()
    X = rfm_full[feat_cols].values.astype(np.float32)
    y_dur = rfm_full["duration"].values.astype(np.float32)
    y_evt = rfm_full["event"].values.astype(np.float32)

    idx_tr, idx_te = train_test_split(
        np.arange(len(X)), test_size=test_size, random_state=random_state, stratify=y_evt
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[idx_tr])
    X_te = scaler.transform(X[idx_te])

    return (
        X_tr.astype(np.float32),
        X_te.astype(np.float32),
        y_dur[idx_tr],
        y_dur[idx_te],
        y_evt[idx_tr],
        y_evt[idx_te],
        scaler,
        idx_tr,
        idx_te,
    )