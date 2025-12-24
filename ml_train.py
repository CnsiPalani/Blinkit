
import os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from src.config import MODEL_PATH, LATE_THRESHOLD_MIN
from src.utils import derive_is_late, feature_engineer, fetch_query


def train_model():
    # Fetch required columns from DB
    df = fetch_query("""
        SELECT order_id, order_date, promised_delivery_time AS promised_time, actual_delivery_time AS actual_time, order_total,
               (SELECT area FROM blinkit_customers c WHERE c.customer_id = o.customer_id) AS region
        FROM blinkit_orders o
        WHERE promised_delivery_time IS NOT NULL AND actual_delivery_time IS NOT NULL
    """)
    df = derive_is_late(df, threshold_min=LATE_THRESHOLD_MIN)
    df = feature_engineer(df)

    df = df.dropna(subset=['is_late'])
    if df.empty:
        raise ValueError("No labeled data found. Please ensure promised_time and actual_time are present.")

    X = df[['hour_of_day', 'day_of_week', 'region']]
    y = df['is_late'].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preproc = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), ['region'])],
                                remainder='passthrough')

    clf = Pipeline([('preproc', preproc),
                    ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))])

    clf.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({'pipeline': clf, 'auc': auc}, MODEL_PATH)
    return auc

if __name__ == '__main__':
    auc = train_model()
    print(f"Model trained. AUC = {auc:.4f}. Saved to {MODEL_PATH}")
